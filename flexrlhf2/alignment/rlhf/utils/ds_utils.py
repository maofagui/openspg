# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
from alignment.util.global_vars import global_context

# DeepSpeed Team
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        dtype="bf16"):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,

        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
        "overlap_comm": True
    }
    zero_opt_dict.update(global_context().runtime_conf.zero_opt_param or {})
    dtype=global_context().runtime_conf.dtype
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        dtype: {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }


def get_eval_ds_config(offload, stage=0, dtype="bf16"):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "overlap_comm": True,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    zero_opt_dict.update(global_context().runtime_conf.zero_opt_param or {})  
    dtype=global_context().runtime_conf.dtype
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        dtype: {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
