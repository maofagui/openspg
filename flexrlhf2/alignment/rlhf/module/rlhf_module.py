# coding: utf-8

import torch
import torch.nn.functional as F
import deepspeed
from alignment.rlhf.utils.utils import print_all_ranks, build_attn_mask
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from alignment.util.global_vars import global_context
from alignment.app.util import logger

def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedRLHFModule:

    def __init__(self, rlhf_engine, tokenizer, config=None):
        """The default RL algorithm is PPO2."""
        self.rlhf_engine = rlhf_engine
        self.tokenizer = tokenizer
        self.config = config

        self.actor_model = self.rlhf_engine.actor
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.critic_model = None

        # self.max_answer_seq_len = self.context.model_conf.max_answer_seq_len
        # if self.tokenizer is not None:
        #     self.end_of_conversation_token_id = self.tokenizer(
        #     context.model_conf.end_of_conversation_token)['input_ids'][-1]

        # # Those value can be changed
        # self.kl_ctl = self.context.model_conf.rl_kl_ctl or 0.02
        # self.clip_reward_value = self.context.model_conf.rl_clip_reward_value or 5
        # self.cliprange = self.context.model_conf.rl_cliprange or 0.2
        # self.cliprange_value = self.context.model_conf.rl_cliprange_value or 0.2
        # self.gamma = self.context.model_conf.rl_gamma or 1.0
        # self.lam = self.context.model_conf.rl_lam or 0.95
        # self.vf_coef = self.context.model_conf.vf_coef or 1.0

        # # used in evaluation
        # self.eval_count = 0
        # self.best_metrics = None
        # self.cur_mode = None

        # # used in updating ref
        # self.trained_steps = 0

        # self.generate_time = 0

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

    def generate_evaluation(self, prompts=None, **kwargs):
        self.eval()
        if prompts is None:
            prompts = kwargs.pop("input_ids")

        max_min_length = self.max_answer_seq_len + prompts.shape[1]
        kwargs["min_length"] = max_min_length
        kwargs["max_length"] = max_min_length

        with torch.no_grad():
            seq = self.actor_model.generate(prompts, **kwargs)
            attention_mask, _ = build_attn_mask(seq, self.tokenizer, self.is_chat_glm)
            prompt_length = prompts.shape[1]
            reward_score = self.reward_model.forward_value(
                seq, attention_mask, prompt_length=prompt_length)['chosen_end_scores'].detach()
        return seq, reward_score

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
            # 如果actor_model是hybrid_engine, 则后续的eval()会打印如下日志(见deepspeed.runtime.hybrid_engine.DeepSpeedHybridEngine.eval)：
            # |E2E latency=57.53s |Gather latency=0.00s (0.00%) |Generate time=56.59s (98.38%) |Training time=0.00s (0.00%) |Others=0.93 (1.62%)|CurSamplesPerSec=0.07 |AvgSamplesPerSec=0.07
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

    def dump_model_norms(self, tag):
        if self.actor_model is not None:
            actor_model_norm = get_model_norm(self.actor_model)
            print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                            self.context.local_rank)
        if self.critic_model is not None:
            critic_model_norm = get_model_norm(self.critic_model)
            print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                            self.context.local_rank)
        if self.ref_model is not None:
            ref_model_norm = get_model_norm(self.ref_model)
            print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                            self.context.local_rank)
        if self.reward_model is not None:
            reward_model_norm = get_model_norm(self.reward_model)
            print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                            self.context.local_rank)

