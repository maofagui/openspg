import torch

from alignment.api.rlhf.config import Placement, AllCollocate, InitialRewardSeparate, ModelRanks
from alignment.api.rlhf.model_provider import ACNoneShareModelProvider, ACNoneShareSEPModelProvider
from alignment.api.utils import dist_util
from alignment.rlhf.data.data_utils import MiniDataset
from alignment.rlhf.distributed.distributed_rlhf_engine import DistributedACEngine
from alignment.rlhf.distributed.model_placement import ModelPlacement
from alignment.rlhf.trainner.app_ds_rlhf_engine import ACNoneShareEngine
from alignment.rlhf.utils.utils import print_rank_0, set_random_seed
from alignment.app.util import logger

hooks = []


def _is_on_policy(ctx):
    return ctx.trainer.rl_policy.lower() == "on"


def next_data(loader):
    for dat in loader:
        yield dat


class ExecutionEngine:
    def __init__(self, 
                 config_path,
                 ):
        """
        :param model_conf:
        :param train_conf:
        :param runtime_conf:
        :param eval_conf:
        :param placement: placement ratio. support three distributed strategies:
        1. Flattening: AllCollocate()
        2. Interleaving: InitialRewardSeparate()
        3. Separation: AllSeparate()
        :param profile_config:
        """
        from hydra import initialize, compose
        with initialize(version_base=None, config_path='../../conf'):
            self.config = compose(config_name=config_path.rsplit('.', 1)[0])

        


        # self.config_path = config_path
        self.actor_model_path = self.config.actor_rollout_ref.model.path
        self.critic_model_path = self.config.critic.model.path

        placement_strategy = self.config.placement.strategy
        if placement_strategy == 'flattening':
            placement = AllCollocate()
        elif placement_strategy == 'interleaving':
            placement = InitialRewardSeparate()
        elif placement_strategy == 'separation':    
            placement = Placement(model_ranks=ModelRanks(
                pred_actor_ranks=self.config.placement.pred_actor_ranks,
                pred_critic_ranks=self.config.placement.pred_critic_ranks,
                init_model_ranks=self.config.placement.init_model_ranks,
                reward_model_ranks=self.config.placement.reward_model_ranks,
                actor_ranks=self.config.placement.actor_ranks,
                critic_ranks=self.config.placement.critic_ranks,
            ))
        self.placement = placement
        # self.placement_ratio = self.config.trainer.placement.ratio
        # placement = {
        #     'flattening': AllCollocate(),
        #     'interleaving':InitialRewardSeparate(),
        #     'separation': AllSeparate()
        # }[placement_strategy]        
        # self.context = global_context()

        self.rl_train_batch_size = self.config.trainer.rl_train_batch_size

        local_rank = dist_util.get_local_rank()
        if local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
        from alignment.app.util.logger import set_log_formatter
        set_log_formatter()
        # setattr(self.context, "local_rank", local_rank)
        self.global_rank = torch.distributed.get_rank()
        
        if self.config.trainer.ac_mode == 'non_share':
            from alignment.rlhf.model.default_model_impl import DefaultACNoneShareModelProvider
            model_provider = DefaultACNoneShareModelProvider(self.actor_model_path, self.critic_model_path)
        print("use model_provider of : {}".format(model_provider))
        num_total_iters = self.config.data.num_total_iters
        # If passed along, set the training seed now.
        set_random_seed(self.config.trainer.seed)
        torch.distributed.barrier()

        # setattr(self.context, "num_self.total_step", num_total_iters)
        logger.info(f"total iteration nums:{num_total_iters}")
        self.context = None
        if isinstance(model_provider, ACNoneShareModelProvider):
            model_placement = ModelPlacement(placement, True, True, True, True)
            self.rlhf_engine = ACNoneShareEngine(
                model_provider=model_provider,
                num_total_iters=num_total_iters,
                config=self.config,
                init_models=False,  # 使用distenv初始化
            )
            from alignment.rlhf.module.ac_none_share_module import ACNoneShareModule
            rl_trainer = ACNoneShareModule
        else:
            raise ValueError(
                "Unknown ACModelProvider type, which should inherit from ACNoneShare-/ACShareModelProvider")
            
        if placement.all_models_flatten():
            self.rlhf_engine.init_all_models()
            self.dist_rlhf_engine = self.rlhf_engine
        else:
            self.dist_rlhf_engine = DistributedACEngine(self.rlhf_engine, model_placement)
            self.rlhf_engine = self.dist_rlhf_engine

        self.trainer = rl_trainer(self.dist_rlhf_engine, None, self.config)
        self.rollout_size = self.config.trainer.rollout_size
        self.batches_per_rollout = self.config.trainer.batches_per_rollout

        self.num_train_epochs = self.config.trainer.num_train_epochs
        self.rl_train_epochs = self.config.trainer.rl_train_epochs
        self.actor_gradient_checkpointing = self.config.trainer.gradient_checkpointing
        num_rollout_in_dataset = 1
        self.reach_max_steps = False

        self.prompt_train_dataloader = [list(range(self.batches_per_rollout)) for _ in range(num_rollout_in_dataset)]

        self.exp_mini_dataset = MiniDataset(self.rollout_size, self.rl_train_batch_size)


    def run(self):
        if self.placement.all_models_separate():
            self._run_separation()
        else:
            self._run_flatten_or_interleaving()
            

    def _run_separation(self):
        num_rollouts = self.context.train_conf.num_rollouts
        rl_train_epochs = self.context.train_conf.rl_train_epochs
        experience_step = 0

        

        rlhf_sep_config = self.context.runtime_conf.rlhf_sep_config

        BATCHES_PER_ROLLOUT = rlhf_sep_config.batches_per_rollout
        ROLLOUT_BATCH_SIZE = rlhf_sep_config.rollout_batch_size

        """对于训练来说，需要训练60(15*5)条数据，
        每次的micro_batch是2，global_batch_size是12，跑5次train。每个micro_batch是2条
        """
        TRAIN_GS_BATCH_SIZE = rlhf_sep_config.train_global_batch_size

        # TRAIN_GS_BATCH_SIZE包含了DP维度，这里就是每次ROLLOUT需要TRAIN多少step
        BATCHES_PER_TRAIN = int(BATCHES_PER_ROLLOUT * ROLLOUT_BATCH_SIZE // TRAIN_GS_BATCH_SIZE)

        for roll_id in range(num_rollouts):
            self.trainer.sync()
            print_rank_0(
                f"Beginning of roll_id {roll_id}/{num_rollouts}, rollout size {self.rollout_size}, Total Generation Batches "
                f"{len(self.prompt_train_dataloader)}", self.global_rank)

            loader = next_data(self.prompt_train_dataloader)

            for _ in range(BATCHES_PER_ROLLOUT):
                out = self.trainer.generate_experience(loader)
            for train_epoch in range(rl_train_epochs):
                self.trainer.set_epoch(train_epoch)
                for _ in range(BATCHES_PER_TRAIN):
                    actor_loss, critic_loss = self.trainer.train_onestep(train_epoch)

        
            
    def _run_flatten_or_interleaving(self):
        for epoch in range(self.num_train_epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch + 1}/{self.num_train_epochs}, rollout size {self.rollout_size}, Total Generation Batches "
                f"{len(self.prompt_train_dataloader)}", self.global_rank)

            self.trainer.eval()
            for step, batch_prompt in enumerate(self.prompt_train_dataloader):
                experience = self.trainer.generate_experience(batch_prompt)
                exp_dataset = self.exp_mini_dataset.add(experience)

                # learn
                if exp_dataset is not None:
                    self.trainer.train()
                    from alignment.rlhf.distributed.distributed_rlhf_engine import SCATTER_QUEUE
                    for i in range(len(SCATTER_QUEUE) - 1, -1, -1):
                        SCATTER_QUEUE[i][0].wait()
                        del SCATTER_QUEUE[i][1]
                        del SCATTER_QUEUE[i]
                    for rl_ep in range(self.rl_train_epochs):
                        for i, exp_data in enumerate(exp_dataset):
                            self.trainer.train_onestep(exp_data)

                        if self.reach_max_steps:
                            break

                    self.trainer.eval()

                if self.reach_max_steps:
                    break
                # end of a batch step of self.prompt_train_dataloader

            if self.reach_max_steps:
                print_rank_0(f"reach max steps {self.context.trainer.max_steps}, end training")
                break
            # end of a epoch of self.num_train_epochs
        if self.exp_mini_dataset.has_remaining():
            print_rank_0(f"Skip training remaining data that is less than roll size.")
            self.exp_mini_dataset.free()