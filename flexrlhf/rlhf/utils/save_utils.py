import os
import re
import torch
import shutil

from alignment.api.rlhf.model_provider import ACShareModelProvider
from alignment.rlhf.utils.utils import print_rank_0, save_hf_format, save_zero_three_model
from alignment.rlhf.module.lora import convert_lora_to_linear_layer
from alignment.rlhf.trainner.app_ds_rlhf_engine import DeepSpeedACShareEngine, DeepSpeedACEngine
from alignment.api.rlhf.config import RestoreConfig


def save_rlhf_checkpoint(directory, rlhf_engine: DeepSpeedACEngine):
    """Saves the current states of the model, optimizer to a folder.
    Only support DeepSpeed distributed type yet. (Others, such as FSDP, MEGATRON_LM, are not implemented)
    """
    # Use the save_checkpoint in DeepSpeedEngine, which will save weights, optimizer, and scheduler.
    # See @ deepspeed.runtime.engine.DeepSpeedEngine.save_checkpoint
    # All other processes need to call save_checkpoint besides rank 0. (but how to collect all ckpts to one folder?)
    from alignment.rlhf.distributed.distributed_rlhf_engine import PatchDistributedEnv

    from alignment.rlhf.distributed.distributed_rlhf_engine import DistributedDeepSpeedACEngine
    if isinstance(rlhf_engine, DistributedDeepSpeedACEngine):
        if rlhf_engine.actor.module:
            with PatchDistributedEnv(current_model=rlhf_engine.actor._current_model):
                rlhf_engine.actor.module.save_checkpoint(directory, tag=RestoreConfig.ACTOR_TAG)
        if hasattr(rlhf_engine, 'critic') and rlhf_engine.critic.module:
            with PatchDistributedEnv(current_model=rlhf_engine.critic._current_model):
                rlhf_engine.critic.module.save_checkpoint(directory, tag=RestoreConfig.CRITIC_TAG)        
    else:                
        rlhf_engine.actor.save_checkpoint(directory, tag=RestoreConfig.ACTOR_TAG)
        if not isinstance(rlhf_engine, DeepSpeedACShareEngine):
            rlhf_engine.critic.save_checkpoint(directory, tag=RestoreConfig.CRITIC_TAG)


def load_rlhf_checkpoint(restore_path, rlhf_engine):
    print_rank_0(f"Load actor checkpoint from {restore_path} with tag {RestoreConfig.ACTOR_TAG}")
    if rlhf_engine.actor.module:
        rlhf_engine.actor.module.load_checkpoint(restore_path, RestoreConfig.ACTOR_TAG)
    if hasattr(rlhf_engine, 'critic') and rlhf_engine.critic.module:
        print_rank_0(f"Load critic checkpoint from {restore_path} with tag {RestoreConfig.CRITIC_TAG}")
        rlhf_engine.critic.module.load_checkpoint(restore_path, RestoreConfig.CRITIC_TAG)



def get_latest_ckpt_step(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return -1
    dirs = os.listdir()
    ckpt_dirs = [i for i in dirs if re.match(r"checkpoint_\d+", i)]
    if len(ckpt_dirs) == 0:
        return -1
    ckpt_dirs.sort()
    return int(ckpt_dirs[-1].split("_")[-1])


def _clear_stale_ckpt(directory, current_dir, keep_ckpt_max=-1):
    if keep_ckpt_max <= 0:
        return
    dirs = os.listdir(directory)
    ckpt_dirs = [i for i in dirs if re.match(r"checkpoint_\d+", i)]
    ckpt_dirs.sort()
    if len(ckpt_dirs) > keep_ckpt_max:
        for ckpt_dir in ckpt_dirs[:-keep_ckpt_max]:
            if current_dir == ckpt_dir:
                continue
            shutil.rmtree(os.path.join(directory, ckpt_dir))


def check_and_save_ckpt(context, step, num_total_step, is_sft=False):
    """If saved ckpt, return True. Return False otherwise.
    """

    # 构造ckpt目录
    ckpt_dir = context.train_conf.checkpoint_dir or context.exporter_conf.deploy_path
    sub_folder = f"checkpoint_{step:0{len(str(num_total_step))}d}"
    directory = os.path.join(ckpt_dir, sub_folder)
    os.makedirs(directory, exist_ok=True)

    # 导出ckpt，并清理旧的目录
    try:
        print_rank_0(f"Saving checkpoint into {directory}")
        if is_sft:
            save_sft_checkpoint(directory, sft_engine=context.sft_engine)
        else:
            save_rlhf_checkpoint(directory, rlhf_engine=context.rlhf_engine)
        _clear_stale_ckpt(ckpt_dir, sub_folder, context.train_conf.keep_checkpoint_max)
        print_rank_0(f"Saved checkpoint into {directory}")
        return True
    except Exception as e:
        print_rank_0(f"Failed to save ckpt {e}")
    return False



def save_rlhf_models(context, rlhf_engine, tokenizer, model_provider):

    print_rank_0('saving model ...')

    ac_share = isinstance(model_provider, ACShareModelProvider)

    output_dir = context.exporter_conf.deploy_path
    actor, critic = None, None
    if rlhf_engine.actor is not None and rlhf_engine.actor.module:
        actor = convert_lora_to_linear_layer(rlhf_engine.actor.module)

    if not ac_share and hasattr(rlhf_engine, 'critic') and \
            rlhf_engine.critic is not None and rlhf_engine.critic.module:

        critic = convert_lora_to_linear_layer(rlhf_engine.critic.module)

    enable_ema = context.model_conf.enable_ema
    if enable_ema:
        rlhf_engine.actor_ema = convert_lora_to_linear_layer(
            rlhf_engine.actor_ema)

    if torch.distributed.get_rank() == 0:
        # TODO: actor后续固定rank0/critic固定rank0有一份 或判断first rank？ && 非share修改测试
        save_hf_format(actor,
                       tokenizer,
                       output_dir,
                       sub_folder='actor')
        if not ac_share:
            save_hf_format(critic,
                           tokenizer,
                           output_dir,
                           sub_folder='critic')
        if enable_ema:
            save_hf_format(rlhf_engine.actor_ema,
                           tokenizer,
                           output_dir,
                           sub_folder='actor_ema')

    actor_conf = context.runtime_conf.actor_conf
    if actor_conf.zero_stage == 3:
        if actor:
            save_zero_three_model(actor,
                                global_rank=context.global_rank,
                                save_dir=os.path.join(output_dir, 'actor'),
                                zero_stage=actor_conf.zero_stage)
        if enable_ema:
            save_zero_three_model(rlhf_engine.actor_ema,
                                  global_rank=context.global_rank,
                                  save_dir=os.path.join(output_dir, 'actor_ema'),
                                  zero_stage=actor_conf.zero_stage)

    critic_conf = context.runtime_conf.critic_conf
    if critic and critic_conf.zero_stage == 3:
        save_zero_three_model(critic,
                              global_rank=context.global_rank,
                              save_dir=os.path.join(output_dir, 'critic'),
                              zero_stage=critic_conf.zero_stage)

