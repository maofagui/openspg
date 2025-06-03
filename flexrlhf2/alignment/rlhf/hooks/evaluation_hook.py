#!/usr/bin/env python
# coding: utf-8
import torch
import time
from tqdm import tqdm

from alignment.util.global_vars import global_context
from alignment.rlhf.hooks.rlhf_train_hook import RLHFTrainHook, SFTTrainHook
from alignment.rlhf.utils.utils import print_rank_0, to_device, get_all_reduce_mean, nested_concat, nested_numpify, distributed_concat

# 存储最近一次评估结果
LATEST_EVAL_RES = {}


class DefaultEvaluationHookImpl:

    def __init__(self, trainer, dataloader, eval_at_first=False, eval_every_steps=None, eval_steps=None):
        self._trainer = trainer
        self._dataloader = dataloader
        self._eval_at_first = eval_at_first
        self._eval_every_steps = eval_every_steps
        self._eval_steps = eval_steps
        self._is_latest_ckpt = False
        self._has_evaluated = False
        self._context = None
        self._eval_count = 0
        self._compute_metric_fn = None
        self._need_wrap_parameter = False

    def on_train_start(self):
        self._context = global_context()
        if self._eval_at_first and not self._has_evaluated:
            self._do_eval(0, 0)

    def on_train_end(self, global_step) -> None:
        """ 整体模型训练结束 """
        if not self._has_evaluated:
            self._do_eval(0, global_step)

    def _do_eval(self, experience_step, global_step):
        pass


class EvaluationHook(DefaultEvaluationHookImpl, RLHFTrainHook):

    def on_experience_train_batch_end(self, experience_step, global_step, metrics=None, **kwargs):
        """评估流程，不与ckpt进行联动"""
        if self._eval_every_steps and global_step % self._eval_every_steps == 0:
            last_mode = self._trainer.cur_mode
            self._do_eval(experience_step, global_step)
            if last_mode == "train":
                self._trainer.train()

    def _visualize_metrics(self, str_prompts, str_outputs, rewards, mean_reward, nth_evaluation, cost_time):
        from rich.table import Table
        from rich.console import Console
        from alignment.rlhf.utils.utils import significant

        print_rank_0("Summarizing evaluation")
        columns = ["prompt", "answer", "reward"]
        columns_data = [str_prompts, str_outputs, rewards]
        rows = list(map(list, zip(*columns_data)))

        # Add metrics/rewards to the table's title
        table_title = f"Evaluation #{nth_evaluation}, mean_reward:{mean_reward}, cost_time:{cost_time}s"
        rich_table = Table(*columns, title=table_title, show_lines=True)
        for ix in range(min(3, len(rows))):
            rich_table.add_row(*[str(significant(x)) for x in rows[ix]])
        Console().print(rich_table)

    def _do_eval(self, experience_step=0, global_step=0):
        """Evaluate"""
        global LATEST_EVAL_RES

        trainer = self._trainer
        context = self._context
        dataloader = self._dataloader

        print_rank_0("Evaluating actor model")
        local_rank = context.local_rank
        device = torch.device("cuda", 0 if local_rank < 0 else local_rank)
        all_samples = []
        all_prompts = []
        all_rewards = []
        cost_time = time.time()

        prompt_size = 0
        total_step = self._eval_steps if self._eval_steps else len(dataloader)
        for i_prompt, batch in tqdm(enumerate(dataloader), total=total_step):
            if "prompt" in batch:
                batch["input_ids"] = batch.pop("prompt")
            if "prompt_att_mask" in batch:
                batch["attention_mask"] = batch.pop("prompt_att_mask")
            batch = to_device(batch, device)
            prompt_size = batch["input_ids"].size()[-1]  # assume the prompt_size is same in all batches

            samples, rewards = trainer.generate_evaluation(**batch)

            all_prompts.extend(batch["input_ids"].tolist())
            all_samples.extend(samples.tolist())
            all_rewards.extend(rewards.tolist())

            if self._eval_steps and i_prompt > self._eval_steps:
                break

        self._has_evaluated = True
        if local_rank <= 0:
            # 解码成文本
            print_rank_0("decoding evaluation results")
            str_prompts = []
            str_outputs = []

            show_lines = 3
            for idx, (prompt_tensor, sample_tensor) in enumerate(zip(all_prompts, all_samples)):
                if idx >= show_lines:
                    break
                str_prompt = trainer.tokenizer.decode(prompt_tensor, skip_special_tokens=True)
                str_output = trainer.tokenizer.decode(sample_tensor[prompt_size:], skip_special_tokens=True)
                str_prompts.append(str_prompt)
                str_outputs.append(str_output)

            mean_reward = torch.tensor(all_rewards).mean().item()
            self._eval_count += 1  # eval_num
            cost_time = time.time() - cost_time

            # Log and display evaluation metrics
            self._visualize_metrics(str_prompts, str_outputs, rewards[:len(str_prompts)], mean_reward,
                                    self._eval_count, cost_time)
            LATEST_EVAL_RES.update({
                "avg_reward": mean_reward,
                "exp_step": experience_step,
                "global_step": global_step
            })

            report_results = {}
            report_results.update(LATEST_EVAL_RES, eval_num=self._eval_count)


class EvaluationSFTHook(DefaultEvaluationHookImpl, SFTTrainHook):

    def on_step_train_end(self, global_step, metrics=None, **kwargs):
        """评估流程，不与ckpt进行联动"""
        if self._eval_every_steps and global_step % self._eval_every_steps == 0:
            self._do_eval(0, global_step)

    def _do_eval(self, experience_step=0, global_step=0):
        """Evaluate, here experience_step is useless"""
        global LATEST_EVAL_RES

        trainer = self._trainer
        context = self._context
        dataloader = self._dataloader

        print_rank_0("Evaluating model")
        local_rank = context.local_rank
        device = torch.device("cuda", 0 if local_rank < 0 else local_rank)
        cost_time = time.time()

        if self._compute_metric_fn is None:
            if context.eval_conf and context.eval_conf.compute_metric_fn:
                from inspect import signature
                sign = signature(context.eval_conf.compute_metric_fn)
                if len(sign.parameters) == 1:
                    self._need_wrap_parameter = True
                    self._compute_metric_fn = context.eval_conf.compute_metric_fn
            else:
                def _inner_compute_fn(losses):
                    try:
                        perplexity = torch.exp(losses)
                    except OverflowError:
                        perplexity = float("inf")

                    try:
                        perplexity = get_all_reduce_mean(perplexity).item()
                        #perplexity = perplexity.mean().item()
                    except:
                        pass
                    return {"perplexity": perplexity}

                self._compute_metric_fn = _inner_compute_fn

        prediction_loss_only = True
        losses_host = None
        preds_host = None
        losses = 0
        for step_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if device:
                batch = to_device(batch, device)
            loss, logits = trainer.prediction_step(batch, prediction_loss_only)
            losses += loss.float()

            if self._eval_steps and step_idx > self._eval_steps:
                break
        losses = losses / (step_idx + 1)
        metrics = self._compute_metric_fn(losses)

        # Log and display evaluation metrics
        print_rank_0(f"metrics: {metrics}, cost_time:{time.time() - cost_time}s")
        metrics.update({"global_step": global_step})
        LATEST_EVAL_RES.update(metrics)

        self._has_evaluated = True

    def _do_eval1(self, experience_step=0, global_step=0):
        """Evaluate, here experience_step is useless"""
        global LATEST_EVAL_RES

        trainer = self._trainer
        context = self._context
        dataloader = self._dataloader

        print_rank_0("Evaluating model")
        local_rank = context.local_rank
        device = torch.device("cuda", 0 if local_rank < 0 else local_rank)
        cost_time = time.time()

        if self._compute_metric_fn is None:
            if context.eval_conf and context.eval_conf.compute_metric_fn:
                from inspect import signature
                sign = signature(context.eval_conf.compute_metric_fn)
                if len(sign.parameters) == 1:
                    self._need_wrap_parameter = True
                    self._compute_metric_fn = context.eval_conf.compute_metric_fn
            else:
                def _inner_compute_fn(losses):
                    try:
                        perplexity = torch.exp(losses.mean())
                    except OverflowError:
                        perplexity = float("inf")

                    try:
                        # perplexity = get_all_reduce_mean(perplexity).item()
                        perplexity = perplexity.item()
                    except:
                        pass
                    return {"perplexity": perplexity}

                self._compute_metric_fn = _inner_compute_fn

        prediction_loss_only = not self._need_wrap_parameter
        losses_host = None
        preds_host = None
        labels_host = None
        for step_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if device:
                batch = to_device(batch, device)
            loss, logits = trainer.prediction_step(batch, prediction_loss_only)
            labels = batch["labels"] if "labels" in batch else None
            _value = [v for v in batch.values()][0]
            batch_size = _value.shape[0]

            if loss is not None:
                # need gather
                losses = distributed_concat(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = distributed_concat(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels,
                                                                               padding_index=-100)
            if logits is not None:
                logits = distributed_concat(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if self._eval_steps and step_idx > self._eval_steps:
                break

        if local_rank <= 0:
            if self._need_wrap_parameter:
                if preds_host is not None:
                    preds_host = nested_numpify(preds_host)
                if labels_host is not None:
                    labels_host = nested_numpify(labels_host)

                from transformers.trainer_utils import EvalPrediction
                metrics = self._compute_metric_fn(
                    EvalPrediction(predictions=preds_host, label_ids=labels_host)
                )
            else:
                metrics = self._compute_metric_fn(losses_host)

            # Log and display evaluation metrics
            print_rank_0(f"metrics: {metrics}, cost_time:{time.time() - cost_time}s")
            metrics.update({"global_step": global_step})
            LATEST_EVAL_RES.update(metrics)

        self._has_evaluated = True
