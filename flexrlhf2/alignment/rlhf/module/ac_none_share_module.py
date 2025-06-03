# coding: utf-8
import time

import torch

from alignment.rlhf.module.rlhf_module import DeepSpeedRLHFModule, gather_log_probs
from alignment.rlhf.distributed.distributed_rlhf_engine import DistributedDeepSpeedACEngine
from alignment.app.util import logger
from alignment.rlhf.config import InitialRewardSeparate

MAPPING_RANKS = dict()


class ACNoneShareDeepSpeedModule(DeepSpeedRLHFModule):
    def __init__(self, rlhf_engine, tokenizer, config):
        super(ACNoneShareDeepSpeedModule, self).__init__(rlhf_engine, tokenizer, config)
        self.critic_model = self.rlhf_engine.critic
        self.run_ref_reward_async = False
        if hasattr(rlhf_engine, 'model_placement') and isinstance(rlhf_engine.model_placement.placement, InitialRewardSeparate):
            self.run_ref_reward_async = True

    def generate_experience(self, data):
        self.eval()
        gen_ret_data = self._generate_sequence(data)
        experience = self._generate_experience_by_seq(gen_ret_data)

        self.train()

        return experience

    def _generate_experience_by_seq(self, gen_ret_data):
        with torch.no_grad():
            action_log_probs = self.actor_model.compute_log_prob(gen_ret_data)

            kwargs = dict(disable_scatter=True) if self.run_ref_reward_async else {}
            if self.ref_model.module is not None or not self.run_ref_reward_async:
                ref_res = self.ref_model.compute_log_prob(gen_ret_data, **kwargs)

            if self.reward_model.module is not None or not self.run_ref_reward_async:
                reward_res = self.reward_model.forward_value(gen_ret_data, **kwargs)

            values = self.critic_model.forward_value(gen_ret_data)

            if self.run_ref_reward_async:
                if self.ref_model.module is not None:
                    ret_data = self.ref_model._all_to_all_single(ref_res, run_async=True)

                if self.reward_model.module is not None:
                    ret_data = self.reward_model._all_to_all_single(reward_res, run_async=True)

                ref_res, reward_res = ret_data[0], ret_data[1]

        experience = dict(base_action_log_probs=action_log_probs,
                          ref_action_log_probs=ref_res,
                          values=values,
                          rewards=reward_res)
        return experience

    def train_onestep(self, inputs):
        self.actor_model.update_actor(inputs)
        self.critic_model.update_critic(inputs)

class ACNoneShareDeepSpeedSEPModule(DeepSpeedRLHFModule):
    def __init__(self, rlhf_engine, tokenizer, context):
        super(ACNoneShareDeepSpeedSEPModule, self).__init__(rlhf_engine, tokenizer, context)
        self._is_sep_model = False
        if hasattr(rlhf_engine, 'pred_actor'):
            self._is_sep_model = True
            self.pred_actor_model = self.rlhf_engine.pred_actor
            self.pred_critic_model = self.rlhf_engine.pred_critic
        self.critic_model = self.rlhf_engine.critic
        self._critic_data_loader = None
        self._actor_data_loader = None
        self._last_e2e_time = None

    def generate_experience(self, prompts):
        self._generate_experience_by_seq(prompts)
        return {}


    def _generate_experience_by_seq(self, prompts):
        forward_use_cache = self.context.model_conf.forward_use_cache
        # seq = seq_dict['all_tokens']
        with torch.no_grad():
            gen_start_time = time.time()
            output = self.pred_actor_model.forward_step(prompts)
            gen_exp_time = time.time() - gen_start_time
            if self.pred_actor_model.replicas[0].module is not None:
                print_throughput_step3_sep(gen_exp_time=gen_exp_time)
            output_ref = self.ref_model.forward_step()
            pred_critic = self.critic_model
            if self._is_sep_model:
                pred_critic = self.pred_critic_model

            values = pred_critic.forward_step()

            reward_score = self.reward_model.forward_step()

        return output, output_ref, reward_score, values

    def train_onestep(self, train_epoch):
        """也可以通过外层set
        """
        last_time = time.time()
        self.actor_model.train_step(train_epoch, self.actor_data_loader)
        actor_train_time = time.time() - last_time
        if self.actor_model.module is not None:
            print_throughput_step3_sep(actor_train_time=actor_train_time)
        
        last_time = time.time()
        self.critic_model.train_step(train_epoch, self.critic_data_loader)
        critic_train_time = time.time() - last_time
        if self.critic_model.module is not None:
            e2e_time = None
            cur_time = time.time()                    
            if self._last_e2e_time is not None:
                e2e_time = cur_time - self._last_e2e_time            
            print_throughput_step3_sep(critic_train_time=critic_train_time, e2e_time=e2e_time)
            self._last_e2e_time = cur_time

        critic_loss, actor_loss = 0., 0.

        return actor_loss, critic_loss

    def _sync_model(self, train_model, pred_model, sync_name, total_layer):
        global_param_dict = {}

        src_model_ranks = train_model._place_policy.all_pipeline_stage_ranks    # train_model，可能多pipeline stage
        from alignment.rlhf.distributed.distributed_rlhf_sep_engine import DistModel
        if isinstance(pred_model, DistModel):
            # pred_critic, interleave_model_parallel_ranks仅包含first/last stage。预测模型目前不开pipe
            dst_model_ranks = [item._place_policy.interleave_model_parallel_ranks[0] for item in pred_model.replicas]    
        else:
            dst_model_ranks = pred_model._place_policy.interleave_model_parallel_ranks

        # logger.info(f'Get src_model_ranks: {src_model_ranks}, dst_model_ranks: {dst_model_ranks}')

        mappings = []
        

        
        for dst_replicate_id, dst_ranks in enumerate(dst_model_ranks):    # 每个dst的model_parallel_rank组都需要copy
            for pipe_stage, src_ranks in enumerate(src_model_ranks): # 每个src_ranks代表一个stage, 可能包含数据并行
                assert len(src_ranks) % len(dst_ranks) == 0, "" # model并行组，src_rank的TP应是dst_rank TP的整数倍
                divisions = len(src_ranks) // len(dst_ranks) # N个src TP对应一个dst TP
                for cur_dst_id, dst_rank in enumerate(dst_ranks):    
                    for src_inner_id in range(divisions):
                        src_rank = src_ranks[cur_dst_id * divisions + src_inner_id]
                        group_ranks = tuple(sorted([src_rank, dst_rank]))
                        if group_ranks not in MAPPING_RANKS:
                            MAPPING_RANKS[group_ranks] = torch.distributed.new_group(group_ranks)
                        mappings.append((src_rank, dst_rank, pipe_stage, src_inner_id, MAPPING_RANKS[group_ranks]))
        # print(mappings)

        src_pipe_stage = len(src_model_ranks)
        layers_per_state = total_layer // src_pipe_stage

        train_model.sync(mappings, {},
                         global_param_dict,
                         group_name=sync_name,
                         layers_per_state=layers_per_state,
                         src_pipe_stage=src_pipe_stage)

        pred_model.sync({},
                        mappings,
                        global_param_dict,
                        group_name=sync_name,
                        layers_per_state=layers_per_state,
                        src_pipe_stage=src_pipe_stage)

    def sync(self):
        
        self.free_data_loader()
        start_time = time.time()
        actor_num_layers = self.context.runtime_conf.rlhf_sep_config.actor_sep_config.num_layers
        self._sync_model(self.critic_model, self.pred_critic_model, sync_name='critic_sync', total_layer=actor_num_layers)

        critic_num_layers = self.context.runtime_conf.rlhf_sep_config.critic_sep_config.num_layers
        self._sync_model(self.actor_model, self.pred_actor_model, sync_name='actor_sync', total_layer=critic_num_layers)
        if self.actor_model.module is not None:
            logger.info(f'Sync cost time: {time.time() - start_time}')

    @property
    def critic_data_loader(self):
        if self._critic_data_loader is None:
            from alignment.rlhf.data.data_utils import SEPMiniDataset

            train_global_batch_size = self.context.runtime_conf.rlhf_sep_config.train_global_batch_size
            train_micro_batch_size = self.context.runtime_conf.rlhf_sep_config.train_micro_batch_size

            critic_dp = len(self.critic_model._place_policy.all_data_parallel_group_ranks[0])
            assert train_global_batch_size % (train_micro_batch_size * critic_dp) == 0

            self._critic_data_loader = SEPMiniDataset(int(train_global_batch_size // critic_dp), train_micro_batch_size)
        
        return self._critic_data_loader
    
    @property
    def actor_data_loader(self):
        if self._actor_data_loader is None:
            from alignment.rlhf.data.data_utils import SEPMiniDataset

            train_global_batch_size = self.context.runtime_conf.rlhf_sep_config.train_global_batch_size
            train_micro_batch_size = self.context.runtime_conf.rlhf_sep_config.train_micro_batch_size

            actor_dp = len(self.actor_model._place_policy.all_data_parallel_group_ranks[0])
            assert train_global_batch_size % (train_micro_batch_size * actor_dp) == 0

            # 单个DP的batch_size
            self._actor_data_loader = SEPMiniDataset(int(train_global_batch_size // actor_dp), train_micro_batch_size)
        
        return self._actor_data_loader
    
    def set_epoch(self, epoch):
        self.actor_data_loader.set_epoch(epoch)
        self.critic_data_loader.set_epoch(epoch)


    def free_data_loader(self):
        self.actor_data_loader.free()
        self.critic_data_loader.free()


class DeepSpeedModuleUnsupervised(ACNoneShareDeepSpeedModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
