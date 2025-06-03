#!/usr/bin/env python
# coding: utf-8
from alignment.util.global_vars import global_context
from alignment.rlhf.hooks.rlhf_train_hook import RLHFTrainHook, SFTTrainHook
from alignment.rlhf.utils.save_utils import check_and_save_ckpt


class DefaultCheckpointSaverHookImpl:
    def on_train_start(self):
        self._context = global_context()
        self._num_total_step = self._context.num_total_step
        self._save_checkpoints_steps = self._context.train_conf.save_checkpoints_steps


class CheckpointSaverHook(DefaultCheckpointSaverHookImpl, RLHFTrainHook):
    def on_experience_train_batch_end(self, experience_step, global_step, metrics=None, **kwargs):
        if global_step % self._save_checkpoints_steps == 0 or global_step > self._num_total_step:
            check_and_save_ckpt(self._context, global_step, self._num_total_step)


class CheckpointSaverSFTHook(DefaultCheckpointSaverHookImpl, SFTTrainHook):

    def on_step_train_end(self, global_step: int, metrics: dict = None) -> None:
        if global_step % self._save_checkpoints_steps == 0 or global_step > self._num_total_step:
            check_and_save_ckpt(self._context, global_step, self._num_total_step, is_sft=True)
