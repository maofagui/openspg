# coding: utf-8

import os
import math
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
)
import functools
from typing import Any, Dict, List, MutableMapping, Tuple, Union

import transformers
from transformers.deepspeed import HfDeepSpeedConfig
from alignment.util.global_vars import global_context
from alignment.app.util import logger

from alignment.rlhf.model.reward_model import RewardModel, load_rm_state_dict_from_ckpt


##### from deepspeed code start

def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False,
                    **hf_config_kwargs):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True,
                                              **hf_config_kwargs)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config, trust_remote_code=True, **hf_config_kwargs)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True,
            **hf_config_kwargs
            )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    zero_stage = 0
    try:
        zero_stage = ds_config["zero_optimization"]["stage"]
    except Exception as e:
        logger.warning(f'Get zero_stage occurs error: {e}')
    logger.info(f'Get critic zero_stage {zero_stage}')        
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)
    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)

    if rlhf_training:
        # critic model needs to load the weight here
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"
        # critic_model.load_state_dict(
        #     torch.load(model_ckpt_path, map_location='cpu'))

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        if not global_context().runtime_conf.skip_load:
            """目前zero3 下还没分shard load ckpt。待修改
            """
            load_rm_state_dict_from_ckpt(critic_model,
                                   model_ckpt_path,
                                   "",
                                   zero_stage=zero_stage)

    return critic_model


create_reward_model = create_critic_model

##### from deepspeed code end


##### from trlx code start
def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def hf_get_decoder(model: nn.Module) -> nn.Module:
    """Returns the causal decoder backbone of the specified HuggingFace transformers
    model.
    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = ("transformer", "model.decoder", "gpt_neox", "decoder")
    return findattr(model, decoder_attrs)


def hf_get_decoder_final_norm(model: nn.Module) -> float:
    """Returns the final (layer) norm of the specified decoder.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
        - transformer.final_layernorm: (ChatGLMModel)
    """
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "model.norm",
        "decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
        "transformer.final_layernorm",
        "final_layernorm",
    )
    return findattr(model, norm_attrs)


def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
        - transformer.layers: (ChatGLMForConditionalGeneration)
        - layers: (ChatGLMModel)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "model.layers",
        "decoder.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
        "transformer.layers",
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_lm_head(model: nn.Module) -> nn.Module:
    """Returns the language modeling (lm) head of the specified HuggingFace
    transformers model.
    NOTE: Different model configurations have different `lm_head` attribute names.
        - lm_head: (GPT2LMHeadModel, BloomForCausalLM)
        - embed_out: (GPTNeoXForCausalLM)
    """
    return model.get_output_embeddings()


def hf_get_hidden_size(config: transformers.PretrainedConfig) -> int:
    """Returns the hidden layer dimensionality of the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different hidden size attribute names.
        - hidden_size: (OPTConfig, BloomConfig)
        - n_embd: (GPT2Config, GPTJConfig)
        - d_model: (PegasusConfig, XLNetConfig)
    """
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


def hf_get_num_hidden_layers(config: transformers.PretrainedConfig) -> int:
    """Returns the number of hidden layers in the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different number-of-layers attribute
    names.
        - num_hidden_layers: (GPTNeoXConfig, OPTConfig)
        - n_layer: (GPT2Config, GPTJConfig, BloomConfig)
    """
    num_hidden_layers_attrs = ("num_hidden_layers", "n_layer")
    return findattr(config, num_hidden_layers_attrs)


def make_head(n_embd: int, out: int, dtype: type = torch.float32) -> nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2, dtype=dtype),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out, dtype=dtype),
    )


def freeze_head_parameters(model, head_attr="v_head"):
    getattr(model, head_attr).requires_grad_(False)


def copy_head_parameter(src_model, target_model, head_attr="v_head"):
    """Copy head parameter from src to target model"""
    src_head = getattr(src_model, head_attr)
    target_head = getattr(target_model, head_attr)
    for source_param, target_param in zip(src_head.parameters(), target_head.parameters()):
        target_param.data.copy_(source_param.data)


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


def freeze_bottom_seq2seq_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    if num_layers_unfrozen == -1:
        return
    shared_embed = model.shared
    decoder_embed = model.decoder.embed_tokens
    encoder_blocks = model.encoder.block
    encoder_norm_layer = model.encoder.final_layer_norm
    decoder_norm_layer = model.decoder.final_layer_norm
    decoder_blocks = model.decoder.block[:-num_layers_unfrozen]
    blocks_to_freeze = (
        list(encoder_blocks)
        + list(decoder_blocks)
        + [shared_embed]
        + [encoder_norm_layer]
        + [decoder_norm_layer]
        + [decoder_embed]
    )
    for block in blocks_to_freeze:
        block.requires_grad_(False)
