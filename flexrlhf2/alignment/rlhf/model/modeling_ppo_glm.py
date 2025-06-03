import os
import gc
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torch.distributed as dist

from copy import deepcopy

from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass

import transformers
from transformers.modeling_outputs import ModelOutput

from trlx.models.modeling_ppo import AutoModelForCausalLMWithValueHead
from trlx.data.method_configs import MethodConfig, register_method

from torchtyping import TensorType
from typing import Any, Dict, List, MutableMapping, Tuple, Union


def hf_get_glm_embeddings(model):
    glm_embeddings_attrs = (
        "glm.word_embeddings",
        "glm.transformer.position_embeddings",
        "glm.transformer.block_position_embeddings",
        "word_embeddings",
        "transformer.position_embeddings",
        "transformer.block_position_embeddings"
    )
    all_embeddings = []
    for attr in glm_embeddings_attrs:
        if rhasattr(model, attr):
            all_embeddings.append(rgetattr(model, attr))
    return all_embeddings



def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []

    glm_embeddings_to_freeze = hf_get_glm_embeddings(model)
    for layer in hidden_layers_to_freeze + glm_embeddings_to_freeze:
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



def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM)
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


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


def hf_get_hidden_size(config: transformers.PretrainedConfig) -> int:
    """Returns the hidden layer dimensionality of the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different hidden size attribute names.
        - hidden_size: (OPTConfig, BloomConfig, GLMConfig)
        - n_embd: (GPT2Config, GPTJConfig)
        - d_model: (PegasusConfig, XLNetConfig)
    """
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


def hf_get_decoder_final_norm(model: nn.Module) -> float:
    """Returns the final (layer) norm of the specified decoder.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
        - glm.transformer.final_layernorm: (GLMForConditionalGeneration)
        - transformer.final_layernorm: (GLMModel)
    """
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
        "glm.transformer.final_layernorm",
        "transformer.final_layernorm",
    )
    return findattr(model, norm_attrs)


# define policy
def dist_fn(p):
    return torch.distributions.Categorical(logits=p)


def flatten_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "/",
) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_tensor_stats(xs: torch.Tensor, mask: torch.Tensor, n: int):
    mean = (xs * mask).sum() / n
    return dict(
        mean=mean,
        min=torch.where(mask.bool(), xs, np.inf).min(),
        max=torch.where(mask.bool(), xs, -np.inf).max(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
    )


def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
        - glm.transformer.layers: (GLMForConditionalGeneration)
        - transformer.layers: (GLMModel)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "decoder.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
        "glm.transformer.layers",
        "glm.word_embeddings",
        "glm.transformer.position_embeddings",
        "glm.transformer.block_position_embeddings",
        "transformer.layers",
        "word_embeddings",
        "transformer.position_embeddings",
        "transformer.block_position_embeddings"
    )
    return findattr(model, hidden_layers_attrs)



@dataclass
@register_method
class PPOMixConfig(MethodConfig):
    """
    Config for PPO method

    :param ppo_epochs: Number of updates per batch
    :type ppo_epochs: int

    :param num_rollouts: Number  of experiences to observe before learning
    :type num_rollouts: int

    :param init_kl_coef: Initial value for KL coefficient
    :type init_kl_coef: float

    :param target: Target value for KL coefficient
    :type target: float

    :param horizon: Number of steps for KL coefficient to reach target
    :type horizon: int

    :param gamma: Discount factor
    :type gamma: float

    :param lam: GAE lambda
    :type lam: float

    :param cliprange: Clipping range for PPO policy loss (1 - cliprange, 1 + cliprange)
    :type cliprange: float

    :param cliprange_value: Clipping range for predicted values
                            (observed values - cliprange_value, observed values + cliprange_value)
    :type cliprange_value: float

    :param vf_coef: Value loss scale w.r.t policy loss
    :type vf_coef: float

    :param gen_kwargs: Additioanl kwargs for the generation
    :type gen_kwargs: Dict[str, Any]

    :param gen_experience_kwargs: if this is not None, then the experience is generated using this
    :type gen_experience_kwargs: Dict[str, Any]

    :param ent_coef: Entropy coefficient for the loss calculation
    :type ent_coef: float
    """

    ppo_epochs: int
    num_rollouts: int
    chunk_size: int
    init_kl_coef: float
    target: float
    horizon: int
    gamma: float
    lam: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    scale_reward: Optional[str]
    ref_mean: Optional[float]
    ref_std: Optional[float]
    cliprange_reward: float
    gen_kwargs: dict
    gen_experience_kwargs: Optional[dict] = None
    ent_coef: float = 0.01
    kl_early_stop: float = None
    clip_ratio: bool = False
    approximate_ratio: bool = False

    def get_advantages_and_returns(
            self,
            values: TensorType["batch_size", "response_size"],
            rewards: TensorType["batch_size", "response_size"],
            response_length: int,
            use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def loss(
            self,
            logprobs: TensorType["batch_size", "response_size"],
            values: TensorType["batch_size", "response_size"],
            old_logprobs: TensorType["batch_size", "response_size"],
            old_values: TensorType["batch_size", "response_size"],
            advantages: TensorType["batch_size", "response_size"],
            returns: TensorType["batch_size", "response_size"],
            mask: TensorType["batch_size", "response_size"],
            logits: TensorType["batch_size", "response_size", "vocabulary_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        n = mask.sum()

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask

        if self.approximate_ratio:
            # using First-order Maclaurin series.
            # ref: https://ai.stackexchange.com/questions/35746/ppo-policy-loss-becomes-nan
            ratio = log_ratio + 1.0
        else:
            ratio = torch.exp(log_ratio)

        # Setting a strict upper & lower bound for ratio
        if self.clip_ratio:
            ratio = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)

        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        # entropy loss
        dist = dist_fn(logits)
        ent_loss = torch.sum(dist.entropy() * mask) / n

        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                policy_loss=pg_loss.item(),
                value_loss=vf_loss.item(),
                entropy=ent_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask) ** 2) / n,
                clipfrac=vf_clipfrac,
            ),
            old_values=get_tensor_stats(old_values, mask, n),
            returns=get_tensor_stats(returns, mask, n),
            policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item()),
            ratio=(ratio * mask).sum() / n,
            padding_percentage=1 - n / mask.numel(),
        )

        return loss, flatten_dict(stats)


@dataclass
@register_method
class PPOSeparateConfig(MethodConfig):
    """
    Config for PPO method

    :param ppo_epochs: Number of updates per batch
    :type ppo_epochs: int

    :param num_rollouts: Number  of experiences to observe before learning
    :type num_rollouts: int

    :param init_kl_coef: Initial value for KL coefficient
    :type init_kl_coef: float

    :param target: Target value for KL coefficient
    :type target: float

    :param horizon: Number of steps for KL coefficient to reach target
    :type horizon: int

    :param gamma: Discount factor
    :type gamma: float

    :param lam: GAE lambda
    :type lam: float

    :param cliprange: Clipping range for PPO policy loss (1 - cliprange, 1 + cliprange)
    :type cliprange: float

    :param cliprange_value: Clipping range for predicted values
                            (observed values - cliprange_value, observed values + cliprange_value)
    :type cliprange_value: float

    :param vf_coef: Value loss scale w.r.t policy loss
    :type vf_coef: float

    :param gen_kwargs: Additioanl kwargs for the generation
    :type gen_kwargs: Dict[str, Any]

    :param gen_experience_kwargs: if this is not None, then the experience is generated using this
    :type gen_experience_kwargs: Dict[str, Any]

    :param ent_coef: Entropy coefficient for the loss calculation
    :type ent_coef: float
    """

    ppo_epochs: int
    num_rollouts: int
    chunk_size: int
    init_kl_coef: float
    target: float
    horizon: int
    gamma: float
    lam: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    scale_reward: Optional[str]
    ref_mean: Optional[float]
    ref_std: Optional[float]
    cliprange_reward: float
    gen_kwargs: dict
    gen_experience_kwargs: Optional[dict] = None
    ent_coef: float = 0.01
    kl_early_stop: float = None
    clip_ratio: bool = False
    approximate_ratio: bool = False

    def get_advantages_and_returns(
            self,
            values: TensorType["batch_size", "response_size"],
            rewards: TensorType["batch_size", "response_size"],
            response_length: int,
            use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0

        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def loss(
            self,
            logprobs: TensorType["batch_size", "response_size"],
            values: TensorType["batch_size", "response_size"],
            old_logprobs: TensorType["batch_size", "response_size"],
            old_values: TensorType["batch_size", "response_size"],
            advantages: TensorType["batch_size", "response_size"],
            returns: TensorType["batch_size", "response_size"],
            mask: TensorType["batch_size", "response_size"],
            logits: TensorType["batch_size", "response_size", "vocabulary_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        n = mask.sum()

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask

        if self.approximate_ratio:
            # using First-order Maclaurin series.
            # ref: https://ai.stackexchange.com/questions/35746/ppo-policy-loss-becomes-nan
            ratio = log_ratio + 1.0
        else:
            ratio = torch.exp(log_ratio)

        # Setting a strict upper & lower bound for ratio
        if self.clip_ratio:
            ratio = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)

        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        # entropy loss
        dist = dist_fn(logits)
        ent_loss = torch.sum(dist.entropy() * mask) / n

        # loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

        stats = dict(
            losses=dict(
                policy_loss=pg_loss.item(),
                value_loss=vf_loss.item(),
                entropy=ent_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask) ** 2) / n,
                clipfrac=vf_clipfrac,
            ),
            old_values=get_tensor_stats(old_values, mask, n),
            returns=get_tensor_stats(returns, mask, n),
            policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item()),
            ratio=(ratio * mask).sum() / n,
            padding_percentage=1 - n / mask.numel(),
        )

        return pg_loss, vf_loss, flatten_dict(stats)


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None
    last_hidden_states: Optional[torch.FloatTensor] = None
    mems: Optional[torch.FloatTensor] = None


class AutoModelForGLMWithValueHead(AutoModelForCausalLMWithValueHead):
    _auto_model_parent_class = transformers.AutoModelForSeq2SeqLM

    def __init__(
            self,
            base_model: transformers.PreTrainedModel,
    ):
        super().__init__(base_model)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            position_ids: Optional[List[torch.FloatTensor]] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True
        forward_kwargs["use_cache"] = use_cache

        outputs = self.base_model.glm(**forward_kwargs)
        value = self.v_head(outputs.last_hidden_states).squeeze(-1)

        if not return_dict:
            outputs = (outputs.logits,) + (value,)
            return outputs

        return CausalLMOutputWithValue(**outputs, value=value)


class AutoModelForGLMWithHydraValueHead(AutoModelForGLMWithValueHead):
    _supported_modules = ["v_head", "frozen_head"]
    _supported_args = ["num_layers_unfrozen"]

    def __init__(
            self, base_model: transformers.PreTrainedModel, *, num_layers_unfrozen: int = -1
    ):
        super().__init__(base_model)
        self.num_layers_unfrozen = num_layers_unfrozen
        if self.num_layers_unfrozen > 0:
            branch_class = GLMModelBranch
            self.frozen_head = branch_class(
                self.base_model,
                num_layers_unfrozen=self.num_layers_unfrozen,
            ).eval()

    def forward_hydra(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            position_ids: Optional[List[torch.FloatTensor]] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return_dict = forward_kwargs.get("return_dict", True)
        forward_kwargs["return_dict"] = True
        forward_kwargs["output_hidden_states"] = True

        outputs = self.forward(**forward_kwargs)
        # Select the hidden state before the first branching layer
        # input_hidden_state = outputs.hidden_states[-(self.num_layers_unfrozen + 1)]
        # output_shape = outputs.hidden_states[-1].size()

        # input_hidden_state = outputs.mems[-(self.num_layers_unfrozen + 1)]
        # mems cache 参考：https://code.alipay.com/ai-dls/antnlp/commit/1e2f40c0110bc22d6336a98cf42266352f684054
        input_hidden_state = outputs.mems[-(self.num_layers_unfrozen)]
        output_shape = outputs.mems[-1].size()

        forward_kwargs.pop("input_ids", None)  # Ignore `input_ids` for branch head
        forward_kwargs.pop(
            "inputs_embeds", None
        )  # Ignore `inputs_embeds` for branch head
        hydra_outputs = self.frozen_head(
            input_hidden_state, output_shape, **forward_kwargs
        )

        if not return_dict:
            return hydra_outputs.logits
        return hydra_outputs


class GLMModelBranch(transformers.PreTrainedModel):
    def __init__(
            self,
            base_model: transformers.PreTrainedModel,
            *,
            num_layers_unfrozen: int,
    ):
        """
        Args:
            base_model (transformers.PreTrainedModel): The pretrained model to extract upper trunk from
            num_layers_unfrozen (int): The number of trainable layers
        """
        super().__init__(base_model.config)

        # The branch is defined by the last `num_layers_unfrozen` layers of the pretrained model
        decoder_blocks = deepcopy(hf_get_decoder_blocks(base_model))
        self.decoder_blocks = nn.ModuleList(list(decoder_blocks)[-num_layers_unfrozen:])
        self.final_norm = deepcopy(hf_get_decoder_final_norm(base_model))
        self.word_embeddings_weight = base_model.glm.word_embeddings.weight
        # self.lm_head = deepcopy(hf_get_lm_head(base_model))

        self.checkpoint_activations = base_model.config.checkpoint_activations

        self.hidden_size = hf_get_hidden_size(self.config)
        self.model_parallel = False
        self.device_map = None
        self.last_device = None
        self.gradient_checkpointing = False

        # Freeze the entire branch
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(  # noqa: max-complexity
            self,
            hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
            output_shape: torch.Tensor,  # output_size given by main trunk
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = False,
            memory_states=None,
    ):
        batch_size, query_length = hidden_states.size()[:2]

        # memory_length = memory_states[0].size(1) if memory_states else 0

        def check_detach(_hidden_states):
            return _hidden_states.detach()

        mem_layers = [check_detach(hidden_states)]
        for i, layer in enumerate(self.decoder_blocks):
            args = [hidden_states, attention_mask]

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs)

                return custom_forward

            mem_i = memory_states[i] if memory_states else None
            if self.checkpoint_activations:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    mem=mem_i,
                )
            else:
                hidden_states = layer(*args, mem=mem_i)
            mem_layers.append(check_detach(hidden_states))
        output = self.final_norm(hidden_states)
        mem_layers = self.update_mems(mem_layers, memory_states)

        logits = F.linear(output, self.word_embeddings_weight)
        return CausalLMOutputWithValue(logits=logits, hidden_states=mem_layers)

    def update_mems(self, hiddens, mems):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length

        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(
                    torch.cat(
                        (mems[i][:, -new_memory_length + query_length:], hiddens[i]),
                        dim=1,
                    )
                )
        return new_mems


class PreTrainedCriticModelWrapper(nn.Module, transformers.utils.PushToHubMixin):
    _auto_model_parent_class: transformers.AutoModel = None
    _supported_modules: List[str] = None
    _supported_args: List[str] = None

    def __init__(
            self, base_model: Optional[transformers.PreTrainedModel] = None, **kwargs
    ):
        super().__init__()
        self.base_model = base_model
        self.forward_kwargs = inspect.getfullargspec(self.base_model.forward).args

    @classmethod
    def _split_kwargs(cls, kwargs: Dict[str, Any]):
        """Separates the kwargs from the supported arguments within `supported_args`
        and those that are not
        """
        supported_kwargs = {}
        unsupported_kwargs = {}
        for key, value in kwargs.items():
            if key in cls._supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value
        return supported_kwargs, unsupported_kwargs

    @classmethod
    def from_pretrained(  # noqa: max-complexity
            cls,
            pretrained_model_name_or_path: Union[str, transformers.PreTrainedModel],
            *model_args,
            **kwargs,
    ):
        """Instantiate a pretrained pytorch model from a pretrained model configuration.
        This method is a wrapper around `transformers.PreTrainedModel.from_pretrained`.
        Please refer to the documentation of `transformers.PreTrainedModel.from_pretrained`
        for more information.

        Args:
            pretrained_model_name_or_path (str):
                The identifier of the pretrained model to load or the pretrained model itself.
            *model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the `_auto_model_parent_class`.
            **kwargs (dict, *optional*):
                Dictionary of keyword arguments to pass to both the underlying `_auto_model_parent_class`
                call (e.g. `transformers.AutoModelForCausalLM.from_pretrained`) and the specific
                instance of the wrapped model.

        NOTE: You must pass in arguments specific to the wrapped model as keyword arguments.
        """
        if kwargs is not None:
            wrapped_model_kwargs, from_pretrained_kwargs = cls._split_kwargs(kwargs)
        else:
            from_pretrained_kwargs = {}
            wrapped_model_kwargs = {}

        if isinstance(pretrained_model_name_or_path, str):
            # Load the base model using the `transformers` AutoClass (e.g. AutoModelForCausalLM)
            base_model = cls._auto_model_parent_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, **from_pretrained_kwargs
            )
        else:
            raise ValueError(
                f"Invalid type for `base_model_name_or_path`: {type(pretrained_model_name_or_path)}"
                "Expected `str`."
            )

        model = cls(base_model, **wrapped_model_kwargs)
        value_head_state_dict = {}

        if os.path.exists(os.path.join(pretrained_model_name_or_path, "head.bin")):
            value_head_state_dict = torch.load(
                os.path.join(pretrained_model_name_or_path, "head.bin"),
                map_location="cpu",
            )

        model.post_init(state_dict=value_head_state_dict)
        return model

    def get_compatible_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Filter out arguments not supported by the specific instance of
        `base_model.transformer.forward`
        """
        # FIXME: This is a hack to get around the fact that the `transformers`
        # architectures we use don't have a consistent API for `forward` parameters.
        return {k: v for k, v in kwargs.items() if k in self.forward_kwargs}

    def post_init(self, *args, **kwargs):
        """Post initialization method. This method is called after the model is
        instantiated and loaded from a checkpoint. It can be used to perform
        additional operations such as loading the state_dict.
        """
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        """Save the pretrained model to a directory. This method is a wrapper
        around `transformers.PreTrainedModel.save_pretrained`. Please refer to
        the documentation of `transformers.PreTrainedModel.save_pretrained` for
        more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `save_pretrained` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `save_pretrained` method.
        """
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        return self.base_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Return the state_dict of the pretrained model."""
        raise NotImplementedError


class AutoModelForCritic(PreTrainedCriticModelWrapper):
    """An `AutoModel` class wrapper for `transformers` causal models that have a
    language modeling head and a value head
    """

    _auto_model_parent_class = transformers.AutoModel
    _supported_modules = ["v_head"]
    _supported_args = ["num_layers_unfrozen"]

    def __init__(
            self,
            base_model: transformers.PreTrainedModel, *, num_layers_unfrozen: int = -1
    ):
        super().__init__(base_model)
        n_embd = (
            base_model.config.hidden_size
            if hasattr(base_model.config, "hidden_size")
            else base_model.config.n_embd
        )
        self.v_head = nn.Linear(n_embd, 1)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            position_ids: Optional[List[torch.FloatTensor]] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"] = True

        outputs = self.base_model(**forward_kwargs)
        value = self.v_head(outputs.last_hidden_states).squeeze(-1)

        if not return_dict:
            outputs = (outputs.logits,) + (value,)
            return outputs

        return CausalLMOutputWithValue(**outputs, value=value)

    def post_init(self, state_dict):
        """
        Adds the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()  # noqa: E702


class AutoModelForGLMSeparate(nn.Module):
    _auto_actor_class = AutoModelForGLMWithHydraValueHead
    _auto_critic_class = AutoModelForCritic

    def __init__(
            self,
            base_model: Optional[transformers.PreTrainedModel] = None,
            critic_model: Optional[transformers.PreTrainedModel] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.critic_model = critic_model

    @classmethod
    def from_pretrained(  # noqa: max-complexity
            cls,
            pretrained_actor_model_path: Union[str, transformers.PreTrainedModel],
            pretrained_critic_model_path: Union[str, transformers.PreTrainedModel],
            *model_args,
            **kwargs,
    ):
        if isinstance(pretrained_actor_model_path, str):
            # Load the base model using the `transformers` AutoClass (e.g. AutoModelForCausalLM)
            actor_model = cls._auto_actor_class.from_pretrained(
                pretrained_actor_model_path, *model_args, **kwargs
            )
        else:
            raise ValueError(
                f"Invalid type for `base_model_name_or_path`: {type(pretrained_actor_model_path)}"
                "Expected `str`."
            )

        if isinstance(pretrained_critic_model_path, str):
            # Load the base model using the `transformers` AutoClass (e.g. AutoModelForCausalLM)
            critic_model = cls._auto_critic_class.from_pretrained(
                pretrained_critic_model_path, *model_args, **kwargs
            )
        else:
            raise ValueError(
                f"Invalid type for `base_model_name_or_path`: {type(pretrained_critic_model_path)}"
                "Expected `str`."
            )

        model = cls(actor_model, critic_model)
        return model

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)