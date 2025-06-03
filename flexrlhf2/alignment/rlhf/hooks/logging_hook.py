#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
from string import Template

from alignment.util.global_vars import global_context
from alignment.rlhf.hooks.rlhf_train_hook import RLHFTrainHook, SFTTrainHook
from alignment.app.util import logger


class LoggingHook(RLHFTrainHook):
    EPOCH_SUMMARY = 'epoch: ${epoch}|rollout_cnt: ${step}|rl_ep: ${rl_ep}|avg_act_loss: ${avg_act_loss}|avg_cri_loss: ${avg_cri_loss}'
    STEP_SUMMARY = 'train_step: ${step}|act_loss: ${actor_loss}|cri_loss: ${critic_loss}'

    def __init__(self, log_steps, rl_epochs, rollout_size):
        self._log_steps = log_steps
        self._rl_epochs = rl_epochs
        self._rollout_size = rollout_size
        self._last_log_step = log_steps
        self._tqdm_bar = tqdm(total=rollout_size, desc="rollout size", delay=1)

    def on_train_start(self):
        self._context = global_context()
        if self._context.local_rank <= 0:
            logger.info(f"{'*' * 20} Start training {'*' * 20}")
            logger.info(f"Config summary:")
            for attr in ["model_conf", "train_conf", "eval_conf", "exporter_conf", "runtime_conf", "profile_conf"]:
                if getattr(self._context, attr, None):
                    logger.info(f"{attr}:{getattr(self._context, attr).to_config(skip_binary=True)}")
                else:
                    logger.info(f"{attr}:None")

    def on_experience_make_start(self, experience_step):
        if self._context.local_rank <= 0:
            logger.info(f"{'*' * 20} rollout#{experience_step} starts {'*' * 20}")
            self._tqdm_bar.reset(total=self._rollout_size)

    def on_experience_batch_start(self, experience_step: int) -> None:
        pass

    def on_experience_batch_end(self, experience_step: int) -> None:
        if self._context.local_rank <= 0:
            self._tqdm_bar.update(1)

    def on_experience_train_start(self, experience_step):
        if self._context.local_rank <= 0:
            logger.info(f"{'*' * 10} rollout#{experience_step} end, train ppo with epoch={self._rl_epochs} {'*' * 10}")

    def on_experience_train_batch_end(self, experience_step, global_step, metrics, **kwargs):
        if self._context.local_rank <= 0 and global_step % self._log_steps == 0:
            metrics.update({"step": global_step})
            logger.info(Template(LoggingHook.STEP_SUMMARY).substitute(**metrics))

    def on_experience_train_end(self, experience_step, global_step, metrics, **kwargs):
        # 这里判断是否满足every_log_steps的规则,需要将原来的取余修改成大于
        step_gap = global_step - self._last_log_step
        if self._context.local_rank <= 0 and step_gap >= self._log_steps:
            logger.info(Template(LoggingHook.EPOCH_SUMMARY).substitute(**metrics))
            logger.info(f"average reward score:{metrics['avg_reward']}")
            self._last_log_step = global_step

    def on_train_end(self, global_step):
        if self._context.local_rank <= 0:
            logger.info(f"{'*' * 20} Training end {'*' * 20}")
