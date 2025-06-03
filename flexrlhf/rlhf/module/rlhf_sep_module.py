# coding: utf-8

import torch.nn.functional as F


from alignment.rlhf.module.rlhf_module import DeepSpeedRLHFModule
class DeepSpeedRLHFSEPModule(DeepSpeedRLHFModule):

    def __init__(self, rlhf_engine, tokenizer, context=None):
        """The default RL algorithm is PPO2."""
        super().__init__(rlhf_engine, tokenizer, context)

        self.pred_actor_model = self.rlhf_engine.pred_actor
        self.pred_critic_model = self.rlhf_engine.pred_critic