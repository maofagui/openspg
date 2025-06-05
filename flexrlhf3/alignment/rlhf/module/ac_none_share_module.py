# coding: utf-8
import time

import torch

from alignment.rlhf.module.rlhf_module import RLHFModule
from alignment.app.util import logger
from alignment.api.rlhf.config import InitialRewardSeparate

MAPPING_RANKS = dict()


class ACNoneShareModule(RLHFModule):
    def __init__(self, rlhf_engine, tokenizer, config):
        super(ACNoneShareModule, self).__init__(rlhf_engine, tokenizer, config)
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
