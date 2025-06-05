# coding: utf-8


class ACEngine:

    def __init__(self, model_provider, num_total_iters, config=None, init_models=True):
        """ Deep speed AC Engine.

        Args:
            model_provider: The model provider.
            num_total_iters: The total number of iterations.
        """
        self.num_total_iters = num_total_iters
        self.model_provider = model_provider
        self.config = config

        if init_models:
            self.init_all_models()
        else:
            self.actor = None
            self.ref = None
            self.reward = None
            self.critic = None

    def init_all_models(self):
        self.actor = self.init_actor()
        if self.config.trainer.enable_ref:
            self.ref = self.init_ref(self.actor)
        self.reward = self.init_reward()

    def init_actor(self):
        return self.model_provider.get_actor_model(None)

    def init_reward(self):
        def _create_reward_model_fn():
            return self.model_provider.get_reward_model()

        # reward_policy = self.config.runtime_conf.reward_policy
        reward_policy = None
        if reward_policy is None:
            reward_engine = _create_reward_model_fn()
        else:
            from alignment.rlhf.trainner.mixed_model import RewardModel
            reward_engine = RewardModel(reward_policy, _create_reward_model_fn, config)

        return reward_engine

    def init_ref(self, actor):

        def _create_ref_model_fn():
            return self.model_provider.get_initial_model()

        # ref_policy = self.config.runtime_conf.ref_policy
        ref_policy = None
        if ref_policy is None:
            ref_engine = _create_ref_model_fn()
        else:
            from alignment.rlhf.trainner.mixed_model import RefModel
            from transformers.models.auto import AutoConfig
            config = AutoConfig.from_pretrained(self.config.model_conf.model_provider.initial_model_path)
            ref_engine = RefModel(ref_policy, _create_ref_model_fn, config)

        return ref_engine


class ACShareEngine(ACEngine):
    """Share actor-critic engine for RLHF"""

    def __init__(self, model_provider, num_total_iters, config=None, init_models=True):
        super().__init__(model_provider, num_total_iters, config, init_models)


class ACNoneShareEngine(ACEngine):

    def __init__(self, model_provider, num_total_iters, config=None, init_models=True):
        super().__init__(model_provider, num_total_iters, config, init_models=init_models)

    def init_all_models(self):
        super(ACNoneShareEngine, self).init_all_models()
        self.critic = self.init_critic()
    
    def init_critic(self):
        return self.model_provider.get_critic_model(None)