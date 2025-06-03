# coding: utf-8
import time

import torch
from alignment.rlhf.module.rlhf_module import DeepSpeedRLHFModule, gather_log_probs
from alignment.rlhf.hooks.profile_train_hook import TraceEventScope
from alignment.rlhf.utils.utils import build_attn_mask, build_position_ids, prepare_inputs_for_generation_glm
from alignment.rlhf.distributed.distributed_rlhf_engine import DistributedDeepSpeedACEngine


# class ACShareTrainModel(TrainModel):
#     def _get_dist_concat_loss(self, losses):
#         actor_loss, critic_loss = losses
#         return torch.concat((actor_loss.unsqueeze(-1), critic_loss.unsqueeze(-1)))

#     def _get_res_loss(self, losses):
#         actor_loss, critic_loss = losses[0], losses[1]
#         return actor_loss, critic_loss


class ACShareDeepSpeedModule(DeepSpeedRLHFModule):
    def __init__(self, rlhf_engine, tokenizer, context):
        super().__init__(rlhf_engine, tokenizer, context)

    def build_forward_params(self, seq, **kwargs):
        if self.is_glm:
            model_inputs = prepare_inputs_for_generation_glm(
                seq,
                position_ids=kwargs["position_ids"],
                generation_attention_mask=kwargs["generation_attention_mask"],
            )
            glm_attention_mask = model_inputs["attention_mask"]
            position_ids = model_inputs["position_ids"]

            return glm_attention_mask, None, position_ids
        else:
            attention_mask, action_mask = build_attn_mask(seq, self.tokenizer, self.is_chat_glm)
            position_ids = build_position_ids(seq, self.tokenizer, self.is_chat_glm)

            return attention_mask, action_mask, position_ids

    def generate_experience(self, prompts, next_prompts=None, **kwargs):
        self.eval()

        experience_step = kwargs.pop('experience_step', 1)

        generate_start = time.time()
        seq, kwargs = self._generate_sequence(prompts, next_prompts, kwargs)
        self.generate_time = time.time() - generate_start
        if seq is None:
            return seq

        from alignment.rlhf.trainner.app_ds_rlhf_trainner import _call_hooks
        _call_hooks(['on_experience_batch_start'], experience_step=experience_step)

        # print(f'Get seq self.tokenizer')

        # decode_prompts = self.tokenizer.batch_decode(prompts.detach().cpu().numpy().tolist())
        # decode_seq = self.tokenizer.batch_decode(seq.detach().cpu().numpy().tolist())
        # print("_generate_sequence prompts is : {} kwargs is : {}, response is : {}".format(decode_prompts, kwargs, decode_seq))
        attention_mask, action_mask, position_ids = self.build_forward_params(seq, **kwargs)

        glm_reward_attention_mask = None
        if self.is_glm:
            glm_reward_attention_mask, action_mask = build_attn_mask(seq, self.tokenizer, self.is_chat_glm)
        forward_kwargs = {}
        if position_ids is not None:
            forward_kwargs["position_ids"] = position_ids

        logits, ref_logits, reward_score, values = self._generate_experience_by_seq(
            attention_mask,
            glm_reward_attention_mask, forward_kwargs, seq)
        values = values[:, :-1]

        # for zero3 dist trace
        # self.train()

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': ref_logits,
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask,
            "action_mask": action_mask,
        }


    def _generate_experience_by_seq(self, attention_mask, glm_reward_attention_mask, forward_kwargs, seq):
        from alignment.app.util import logger
        logger.info("_generate_experience_by_seq attention_mask is : {}, glm_reward_attention_mask : {}".format(
            attention_mask.shape if attention_mask is not None else None, glm_reward_attention_mask.shape if glm_reward_attention_mask is not None else None))
        forward_use_cache = self.context.model_conf.forward_use_cache

        with torch.no_grad():
            frozen_layer_conf = self.context.model_conf.frozen_layer_conf
            run_ref_reward_async = False
            if isinstance(self.rlhf_engine,
                          DistributedDeepSpeedACEngine) and self.context.model_conf.run_ref_reward_async:
                run_ref_reward_async = True

            with TraceEventScope('forward_ref'):
                if frozen_layer_conf:
                    ref_res = self.actor_model.forward_hydra(seq,
                                                                        attention_mask=attention_mask,
                                                                        return_dict=True,
                                                                        use_cache=forward_use_cache,
                                                                        **forward_kwargs)
                else:
                    if self.ref_model.module is not None or not run_ref_reward_async:
                        ref_res = self.ref_model(
                            seq,
                            attention_mask=attention_mask,
                            use_cache=forward_use_cache,
                        # deepspeed chat跑不了
                            return_dict=True,
                            **forward_kwargs)
                ## 拿到ref_logits
                # ref_logits = ref_logits.to(seq.device)

            # output_ref = self.ref_model(seq, attention_mask=attention_mask)

            if self.reward_model.module is not None or not run_ref_reward_async:
                if glm_reward_attention_mask is None:
                    glm_reward_attention_mask = attention_mask 
                with TraceEventScope('forward_reward'):
                    reward_res = self.reward_model.forward_value(seq,
                                                                 glm_reward_attention_mask,
                                                                 use_cache=forward_use_cache,
                                                                 prompt_length=self.prompt_length)

            if run_ref_reward_async:
                bs, seq_len = seq.shape[0:2]
                seq_len -= 1

                if self.ref_model.module is not None:
                    prev_idx, cur_len = 0, bs * (seq_len + 1)
                    output_tensor = self.ref_model.tmp_empty_tensor.narrow(0, prev_idx, cur_len).view(-1)

                    prev_idx += cur_len
                    cur_len = bs * seq_len * 2
                    input_tensor = self.ref_model.tmp_empty_tensor.narrow(0, prev_idx, cur_len).view((bs * 2, seq_len))
                    input_tensor = torch.cat(ref_res, out=input_tensor).view(-1)

                    del ref_res
                    self.ref_model._all_to_all_single(input_tensor,
                                                      output_tensor,
                                                      input_split_sizes=[bs * seq_len, bs * seq_len],
                                                      output_split_sizes=[bs * seq_len, bs],
                                                      run_async=True)

                if self.reward_model.module is not None:
                    prev_idx, cur_len = 0, bs * (seq_len + 1)
                    output_tensor = self.reward_model.tmp_empty_tensor.narrow(0, prev_idx, cur_len).view(-1)

                    prev_idx += cur_len
                    cur_len = bs * 2
                    input_tensor = self.reward_model.tmp_empty_tensor.narrow(0, prev_idx, cur_len).view((bs * 2))
                    input_tensor = torch.cat(reward_res, out=input_tensor).view(-1)
                    del reward_res

                    self.reward_model._all_to_all_single(input_tensor,
                                                         output_tensor,
                                                         input_split_sizes=[bs, bs],
                                                         output_split_sizes=[bs * seq_len, bs],
                                                         run_async=True)

                reward_score = output_tensor[bs * seq_len:].view((bs))
                ref_logits = output_tensor[0:bs * seq_len].view((bs, seq_len))

            else:
                ref_logits = ref_res.logits
                reward_score = reward_res['chosen_end_scores']
            if len(ref_logits.shape) == 2:
                ref_logprobs = ref_logits
            else:
                ref_logprobs = gather_log_probs(ref_logits[:, :-1, :], seq[:, 1:])

            # for zero3 trace, actor_model forward放最后计算。(hybrid时, ref也会引用到actor的base_model)
            with TraceEventScope('forward_actor'):
                logits, *_, values = self.actor_model(seq,
                                                      attention_mask=attention_mask,
                                                      **forward_kwargs,
                                                      use_cache=forward_use_cache)
        return logits, ref_logprobs, reward_score, values

    def train_onestep(self, inputs, next_inputs):
        @TraceEventScope("train_actor")
        @ACShareTrainModel(self.actor_model, self.rlhf_engine)
        def train_actor(inputs, next_inputs):
            # train the rlhf mode here
            ### process the old outputs
            prompts = inputs['prompts']
            log_probs = inputs['logprobs']
            ref_log_probs = inputs['ref_logprobs']
            reward_score = inputs['rewards']
            values = inputs['value']
            attention_mask = inputs['attention_mask']
            seq = inputs['input_ids']
            # print(f'after rank: {torch.distributed.get_rank()}, {inputs["rewards"]} {inputs["ref_logprobs"]}')
            """
            print("ppo_debug prompts is : {}".format(prompts))
            print("ppo_debug log_probs is : {}".format(log_probs))
            print("ppo_debug ref_log_probs is : {}".format(ref_log_probs))
            print("ppo_debug reward_score is : {}".format(reward_score))
            print("ppo_debug values is : {}".format(values))
            """
            action_mask = inputs['action_mask']

            start = prompts.size()[-1] - 1

            old_values = values
            with torch.no_grad():
                old_rewards = self.compute_rewards(prompts, log_probs,
                                                   ref_log_probs, reward_score,
                                                   action_mask)
                advantages, returns = self.get_advantages_and_returns(
                    old_values, old_rewards, start)

            ### process the new outputs
            batch = {'input_ids': seq, "attention_mask": attention_mask}
            ## TODO 也对齐。。。。
            with TraceEventScope('forward_actor'):
                outputs = self.actor_model(**batch, use_cache=False, return_dict=True)

            actor_prob = outputs.logits
            value = outputs.value[:, :-1]
            actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], inputs['input_ids'][:, 1:])
            actor_loss = self._actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages,
                                             action_mask[:, start:])  # pg_loss

            # 计算 vf_loss
            critic_loss = self._critic_loss_fn(value[:, start:], old_values[:, start:],
                                               returns, action_mask[:, start:])
            total_loss = actor_loss + self.vf_coef * critic_loss

            with TraceEventScope('backward_actor'):
                self.actor_model.backward(total_loss)
            return actor_loss, critic_loss

        actor_loss, critic_loss = train_actor(inputs, next_inputs)
        # TODO 原来trlx的实现，是按mini-batch进行梯度累加的，num_mb个mini-batch后step()一次，并加一次scheduler.step()
        # 这里是每个batch更新一次梯度。
        with TraceEventScope('step_actor'):
            self.actor_model.step()
        # 有些RL算法需要更新ref
        self.trained_steps += 1
        self.update_ref_model()

        return actor_loss, critic_loss

    def update_ref_model(self):
        if hasattr(self.context.model_conf, "update_ref") and self.context.model_conf.update_ref:
            if self.trained_steps % self.context.model_conf.update_ref_per_step == 0:
                frozen_layer_conf = self.context.model_conf.frozen_layer_conf
                ref_model_with_head = self.ref_model.frozen_head if frozen_layer_conf else self.ref_model
                copy_head_parameter(self.actor_model, ref_model_with_head)


class ACShareDeepSpeedPPO2Module(ACShareDeepSpeedModule):
    pass
