# coding: utf-8

import time
import os
import torch
import random
import numpy as np
import math
import deepspeed

from transformers import set_seed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from typing import Mapping, Union, Any, Optional
from numbers import Number


def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def significant(x: Number, ndigits=2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or math.isnan(x) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def save_hf_format(model, tokenizer, output_dir, sub_folder="", ):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model

    from alignment.rlhf.model.modeling_ppo import PreTrainedModelWrapper
    if isinstance(model_to_save, PreTrainedModelWrapper):
        print_rank_0('only saving the base model for wrapped model')
        model_to_save = model_to_save.base_model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(output_dir, sub_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)

    if hasattr(model_to_save, 'config'):
        model_to_save.config.to_json_file(output_config_file)
    elif hasattr(model_to_save, 'base_model'):
        model_to_save.base_model.config.to_json_file(output_config_file)

    tokenizer.save_vocabulary(output_dir)
    print_rank_0(f'saved hf format to {output_dir}')


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=["bias", "LayerNorm.weight"]):
    """将model中的需要计算gradient的parameters, 按照变量是否在no_decay_name_list分成两个group，一个加weight_decay, 另一个不加"""
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    from alignment.rlhf.model.modeling_ppo import PreTrainedModelWrapper
    if isinstance(model_to_save, PreTrainedModelWrapper):
        print_rank_0('only saving the base model for wrapped model')
        model_to_save = model_to_save.base_model

    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def get_dynamic_port(custom_port=False):
    from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
    if custom_port is None or (isinstance(custom_port, bool) and not custom_port):
        return TORCH_DISTRIBUTED_DEFAULT_PORT
    elif isinstance(custom_port, bool) and custom_port:
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        devices_idx = int(devices.split(",")[0])
        return TORCH_DISTRIBUTED_DEFAULT_PORT + devices_idx
    elif isinstance(custom_port, int) or isinstance(custom_port, str):
        return int(custom_port)
    else:
        raise ValueError(f"Unknown custom_port:{custom_port}, please using bool or int")


def build_attn_mask(input_ids, tokenizer, is_chat_glm=False):
    if is_chat_glm:
        assert tokenizer.padding_side == "left"
    return _build_attn_mask(input_ids, tokenizer.pad_token_id, tokenizer.bos_token_id, is_chat_glm)


def _build_attn_mask(input_ids, pad_token_id, bos_token_id=None, is_chat_glm=False):
    """
    Ref: https://huggingface.co/THUDM/chatglm-6b/blob/main/tokenization_chatglm.py
    """

    def inner_build_attn_mask(_input_ids, line_break_token_id=4):
        c_indices = (_input_ids == pad_token_id).nonzero()
        c_ind = c_indices[-1].item() + 1 if len(c_indices) > 0 else 0
        # 将最后的换行符，当做pad处理
        for i in range(len(_input_ids)):
            if _input_ids[-i - 1] != line_break_token_id:
                break
        d_ind = i
        seq_length = _input_ids.shape[-1] - c_ind - d_ind
        if bos_token_id in _input_ids:
            context_length = (_input_ids == bos_token_id).nonzero()[0].item()
            context_length -= c_ind
        else:
            context_length = seq_length
        attention_mask = np.ones((1, seq_length, seq_length))
        attention_mask = np.tril(attention_mask)
        attention_mask[:, :, :context_length] = 1
        attention_mask = np.bool_(attention_mask < 0.5)
        # padding left
        if c_ind > 0:
            attention_mask = np.pad(attention_mask,
                                    pad_width=[(0, 0), (c_ind, 0), (c_ind, 0)],
                                    mode='constant', constant_values=True)
        # padding right
        if d_ind > 0:
            attention_mask = np.pad(attention_mask,
                                    pad_width=[(0, 0), (0, d_ind), (0, d_ind)],
                                    mode='constant', constant_values=True)

        action_mask = ~attention_mask[0][-d_ind - 1, :-1]
        return attention_mask, action_mask

    if is_chat_glm:
        if len(input_ids.shape) > 1:
            bs = input_ids.shape[0]
            attention_mask_np = []
            action_mask_np = []
            for idx in range(bs):
                attention_mask_, action_mask_ = inner_build_attn_mask(input_ids[idx])
                attention_mask_np.append(attention_mask_)
                action_mask_np.append(action_mask_)
        else:
            attention_mask_np, action_mask_np = inner_build_attn_mask(input_ids)
        attention_mask = torch.tensor(attention_mask_np)
        action_mask = torch.tensor(action_mask_np)
    else:
        attention_mask = input_ids.not_equal(pad_token_id).long()
        action_mask = attention_mask[:, 1:]
    return attention_mask.to(input_ids.device), action_mask.to(input_ids.device)


def _build_position_ids(input_ids, pad_token_id, bos_token_id, mask_token_id, gmask_token_id,
                        line_break_token_id=4):
    """
    Ref: https://huggingface.co/THUDM/chatglm-6b/blob/main/tokenization_chatglm.py
    """

    def inner_build_position_ids(_input_ids):
        c_indices = (_input_ids == pad_token_id).nonzero()
        c_ind = c_indices[-1].item() + 1 if len(c_indices) > 0 else 0
        # 将最后的换行符，当做pad处理。
        for i in range(len(_input_ids)):
            if _input_ids[-i - 1] != line_break_token_id:
                break
        d_ind = i
        seq_length = _input_ids.shape[-1] - c_ind - d_ind
        if bos_token_id in _input_ids:
            context_length = (_input_ids == bos_token_id).nonzero()[0].item()
            context_length -= c_ind
        else:
            context_length = seq_length
        position_ids = np.arange(seq_length, dtype=np.int64)
        mask_token = mask_token_id if mask_token_id in _input_ids else gmask_token_id
        if mask_token in _input_ids:
            mask_position = (_input_ids == mask_token).nonzero()[0].item()
            mask_position -= c_ind
            position_ids[context_length:] = mask_position
        block_position_ids = np.concatenate(
            [np.zeros(context_length, dtype=np.int64),
             np.arange(1, seq_length - context_length + 1, dtype=np.int64)])
        position_ids = np.stack([position_ids, block_position_ids], axis=0)
        if c_ind > 0:
            position_ids = np.pad(position_ids, pad_width=[(0, 0), (c_ind, 0)])
        if d_ind > 0:
            position_ids = np.pad(position_ids, pad_width=[(0, 0), (0, d_ind)])
        return position_ids

    if len(input_ids.shape) > 1:
        bs = input_ids.shape[0]
        res_np = []
        for idx in range(bs):
            res_np.append(inner_build_position_ids(input_ids[idx]))
    else:
        res_np = inner_build_position_ids(input_ids)
    return torch.tensor(res_np)


def build_position_ids(input_ids, tokenizer, is_chat_glm=False):
    if not is_chat_glm:
        return None
    bos_token_id = tokenizer.bos_token_id
    gmask_token_id = tokenizer.gmask_token_id
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    position_ids = _build_position_ids(input_ids, pad_token_id, bos_token_id, mask_token_id, gmask_token_id)
    position_ids = position_ids.to(input_ids.device)
    return position_ids


def formatted_print_tokenizer(tokenizer):
    properties = ["name_or_path", "tokenizer_class", "padding_side"]
    tokens = ["bos_token", "bos_token_id",
              "eos_token", "eos_token_id",
              "end_token", "end_token_id",
              "pad_token", "pad_token_id",
              "unk_token", "unk_token_id"]
    extra_tokens = ["gmask_token", "gmask_token_id",
                    "mask_token", "mask_token_id"]
    print("--" * 10 + "tokenizer attrs" + "--" * 10)
    for attr in properties + tokens + extra_tokens:
        if hasattr(tokenizer, attr):
            print(f"{attr}: {getattr(tokenizer, attr)}")
    print("--" * 20)


def prepare_inputs_for_generation_glm(
        input_ids, past=None, position_ids=None, generation_attention_mask=None, **kwargs
):
    # only last token for inputs_ids if past is defined in kwargs
    attention_mask = generation_attention_mask
    seq_length = input_ids.shape[1]
    if past:
        if position_ids is not None:
            position_ids = position_ids[:, :, seq_length - 1].unsqueeze(-1)
        if attention_mask is not None:
            attention_mask = attention_mask[
                             :, :, seq_length - 1, :seq_length
                             ].unsqueeze(-2)
        input_ids = input_ids[:, -1].unsqueeze(-1)
    else:
        if position_ids is not None:
            position_ids = position_ids[:, :, :seq_length]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :seq_length, :seq_length]
    if position_ids is not None and input_ids.size(0) > position_ids.size(0):
        batch_size = position_ids.size(0)
        num_beams = input_ids.size(0) // batch_size
        position_ids = position_ids.unsqueeze(1).expand(-1, num_beams, -1, -1)
        position_ids = position_ids.reshape(
            batch_size * num_beams, *position_ids.shape[-2:]
        )
    if attention_mask is not None and input_ids.size(0) > attention_mask.size(0):
        batch_size = attention_mask.size(0)
        num_beams = input_ids.size(0) // batch_size
        attention_mask = attention_mask.unsqueeze(1).expand(-1, num_beams, -1, -1, -1)
        attention_mask = attention_mask.reshape(
            batch_size * num_beams, *attention_mask.shape[-3:]
        )
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "mems": past,
    }


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()

def nested_detach(tensors):
    """Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()


def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0]:, : tensor2.shape[1]] = tensor2
    return result


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0]:, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def distributed_concat(tensor: Any, num_total_examples: Optional[int] = None) -> Any:
    import torch.distributed as dist
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
        if isinstance(tensor, Mapping):
            return type(tensor)({k: distributed_concat(t, num_total_examples) for k, t in tensor.items()})
        tensor = atleast_1d(tensor).contiguous()
        output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")
