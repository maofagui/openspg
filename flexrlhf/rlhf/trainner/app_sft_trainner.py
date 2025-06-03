# coding: utf-8

import inspect
import torch
import deepspeed

from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer

from alignment.util.global_vars import global_context
from alignment.api.rlhf.config import ModelConfig, TrainConfig, EvalConfig, ExporterConfig, RuntimeConfig, \
    Placement, AllCollocate, ProfilerConfig
from alignment.api.utils import dist_util
from alignment.rlhf.data.default_data_impl import create_datasets
from alignment.rlhf.hooks import LoggingSFTHook
from alignment.rlhf.module.sft_module import SFTTrainModule
from alignment.rlhf.trainner.app_ds_rlhf_engine import DeepSpeedSFTEngine
from alignment.rlhf.utils.save_utils import load_sft_checkpoint
from alignment.rlhf.utils.utils import print_rank_0, to_device, set_random_seed, \
    get_dynamic_port
from alignment.app.util import logger

hooks = []


def _call_hooks(func, **kwargs):
    if isinstance(func, str):
        func = [func]
    for hook in hooks:
        try:
            for f in func:
                getattr(hook, f)(**kwargs)
        except:
            logger.error(f"Failed to call {hook}.{f} with {kwargs}")
            raise


def train_sft(model_conf: ModelConfig,
              train_conf: TrainConfig,
              runtime_conf: RuntimeConfig,
              eval_conf: Optional[EvalConfig] = None,
              exporter_conf: Optional[ExporterConfig] = None,
              placement: Placement = AllCollocate(),
              profile_config: ProfilerConfig = None):
    # 将参数存储到上下文
    context = global_context()
    setattr(context, "model_conf", model_conf)
    setattr(context, "train_conf", train_conf)
    setattr(context, "eval_conf", eval_conf)
    setattr(context, "exporter_conf", exporter_conf)
    setattr(context, "runtime_conf", runtime_conf)
    setattr(context, "profile_conf", profile_config)

    local_rank = dist_util.get_local_rank()
    if local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        deepspeed.init_distributed(distributed_port=get_dynamic_port(runtime_conf.custom_port))

    setattr(context, "local_rank", local_rank)
    global_rank = torch.distributed.get_rank()

    # 设置seed;等待其他机器启动
    set_random_seed(runtime_conf.seed)
    torch.distributed.barrier()

    # 初始化tokenizer
    assert model_conf.model_provider, "Missing model_provider in model config."
    model_provider = model_conf.model_provider
    if model_provider.tokenizer:
        tokenizer = model_provider.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_provider.initial_model_path,
                                                  fast_tokenizer=True)
        tokenizer.pad_token = tokenizer.eos_token

    # 构建train dataloader
    if isinstance(train_conf.data_conf.dataset, DataLoader):
        train_dataloader = train_conf.data_conf.dataset
    elif inspect.isfunction(train_conf.data_conf.dataset):
        # 函数用法参考 alignment.rlhf.data.data_utils.instantiate_dataloader
        train_dataloader = train_conf.data_conf.dataset(*train_conf.data_conf.dataset_args)
    else:
        data_provider = create_datasets
        if train_conf.data_conf.data_range is None:
            train_conf.data_conf.data_range = (0, 1.0)
        train_dataloader, _ = \
            data_provider(context, tokenizer, train_phase=1, data_conf=train_conf.data_conf)

    # 构建eval dataloader
    eval_dataloader = None
    if eval_conf and eval_conf.data_conf:
        eval_data_conf = eval_conf.data_conf
        if isinstance(eval_data_conf.dataset, DataLoader):
            eval_dataloader = eval_data_conf.dataset
        elif inspect.isfunction(eval_data_conf.dataset):
            eval_dataloader = eval_data_conf.dataset(*eval_data_conf.dataset_args)
        else:
            if eval_data_conf.data_range is None:
                eval_data_conf.data_range = (0, 1.0)
            eval_dataloader, _ = create_datasets(context, tokenizer, train_phase=1, data_conf=eval_data_conf)

    # 初始化训练engine
    num_total_iters = len(train_dataloader) * train_conf.data_conf.batch_size * \
                      context.train_conf.num_train_epochs / \
                      context.train_conf.optimizer.gradient_accumulation_steps
    sft_engine = DeepSpeedSFTEngine(model_conf.model_provider, num_total_iters, context, init_models=True)

    setattr(context, "num_total_step", num_total_iters)
    setattr(context, "sft_engine", sft_engine)

    # restore deepspeed ckpt
    if context.runtime_conf.restore_conf:
        load_sft_checkpoint(context.runtime_conf.restore_conf.restore_path, sft_engine)

    trainer = SFTTrainModule(sft_engine, tokenizer, context)

    if eval_dataloader:
        from alignment.rlhf.hooks.evaluation_hook import EvaluationSFTHook
        hooks.append(EvaluationSFTHook(trainer, eval_dataloader, context.eval_conf.eval_at_first,
                                       context.eval_conf.eval_every_steps, context.eval_conf.steps))

    # Train!
    total_step = 0
    reach_max_steps = False
    _call_hooks(['on_train_start'])

    ds_conf = runtime_conf.dist_conf
    if ds_conf.gradient_checkpointing:
        sft_engine.model.gradient_checkpointing_enable()

    for epoch in range(train_conf.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{train_conf.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            global_rank)
        for step, batch in enumerate(train_dataloader):
            trainer.train()
            _call_hooks(['on_step_train_start'], global_step=total_step)
            batch = to_device(batch, device)

            loss = trainer.do_train(inputs=batch, use_cache=False)

            total_step += 1
            _call_hooks(['on_step_train_end'], global_step=total_step, metrics={"loss": loss.item()})
            reach_max_steps = 0 < context.train_conf.max_steps < total_step
            if reach_max_steps:
                break

        sft_engine.model.tput_timer.update_epoch_count()
        if reach_max_steps:
            print_rank_0(f"reach max steps {context.train_conf.max_steps}, end training")
            break

    _call_hooks('on_train_end', global_step=total_step)
