# coding: utf-8

import torch

class RLHFModule:

    def __init__(self, rlhf_engine, tokenizer, config=None):
        """The default RL algorithm is PPO2."""
        self.rlhf_engine = rlhf_engine
        self.tokenizer = tokenizer
        self.config = config

        self.actor_model = self.rlhf_engine.actor
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.critic_model = None

    @property
    def is_chat_glm(self):
        return self.context.model_conf.model_arch_type == "chatglm"

    @property
    def is_glm(self):
        return self.context.model_conf.model_arch_type == "glm"

    def _generate_sequence(self, data):
        with torch.no_grad():
            seq = self.actor_model.generate(data)
        return seq

    def generate_experience(self, prompts, **kwargs):
        pass

    def train_onestep(self, inputs, **kwargs):
        pass

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[-1] - 1
        # mask按照dim=1进行sum，有mask的位置(即pad)为1，求和后能得到pad的总数，推算出reward的位置
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        last_gaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gaelam = delta + self.gamma * self.lam * last_gaelam
            advantages_reversed.append(last_gaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        if self.actor_model is not None:
            assert self.actor_model.module.training
        if self.critic_model is not None:
            assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        if self.actor_model is not None:
            assert not self.actor_model.module.training
        if self.critic_model is not None:
            assert not self.critic_model.module.training
        if self.ref_model is not None:
            assert not self.ref_model.module.training
        if self.reward_model is not None:
            assert not self.reward_model.module.training

    def train(self):
        if self.actor_model is not None:
            self.actor_model.train()
        if self.critic_model is not None:
            self.critic_model.train()
        self.cur_mode = "train"

    def eval(self):
        if self.actor_model is not None:
            self.actor_model.eval()
        if self.critic_model is not None:
            self.critic_model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        self.cur_mode = "eval"

    def _actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def _critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss