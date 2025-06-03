# coding: utf-8

import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):
    """Support GPT-J/Llama/ChatGLM model and tokenizer"""

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, extract_transformer=False,
                 score_strategy="last", head_dim=1):
        """
        Args:
            extract_transformer(bool): If True, will extract transformer out from base_model as self.rm_transformer.
                If the base_model is created by AutoModel, set extract_transformer to False.
                If the original base_model is task-specific (such as AutoModelForCausalLM), set it to True.
            score_strategy: the strategy to pick out the score from the logits of v_head. The loss is different when
                using different strategy.
                - divergence (default): use the logits of v_head whose index is the last not-pad index in input_ids.
                    the loss is calculated by the mean logits of divergent chosen- and rejected- rewards.
                - last: use the logits of v_head whose index is just the last one.
                    The loss is calculated by the mean logits of all chosen- and rejected- rewards.
                    (recommended for glm, for its higher accuracy than divergence's)
        Note:
            For llama 1, the last dimension of n_embd is same as the vocab_size, not the hidden_size
        """
        super().__init__()
        self.config = base_model.config
        self.is_chat_glm = self.config.architectures[0] == "ChatGLMModel"
        self.is_llama = "llama" in self.config.architectures[0].lower()
        self.num_padding_at_beginning = num_padding_at_beginning
        self.head_dim = head_dim
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            n_embd = self.config.word_embed_proj_dim
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            n_embd = self.config.n_embd
        self.v_head = nn.Linear(n_embd, self.head_dim, bias=False)
        # 如果base_model模型是通过AutoModelForCausalLM.from_pretrained创建，会包含transformer和lm_head两部分。
        # 这里只需用transformer部分(但为了兼容ds-chat的模型，需要手动设置extract_transformer=True）。
        # 如果用包含lm_head的base_model，在load_state_dict时，可能会遇到key不匹配的问题。
        if extract_transformer and hasattr(base_model, "transformer"):
            self.rm_transformer = base_model.transformer
        else:
            self.rm_transformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

        # 如果prompt的pad在左边，那么在forward要注意处理。
        self.is_pad_left = tokenizer.padding_side == 'left' if hasattr(tokenizer, "padding_side") else False
        self.score_strategy = score_strategy

    def enable_input_require_grads(self):
        self.rm_transformer.enable_input_require_grads()

    def gradient_checkpointing_enable(self):
        self.rm_transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rm_transformer.gradient_checkpointing_disable()

    def _move_pad_to_end(self, ind_tensor, rewards):
        not_pad_ind = (ind_tensor != self.PAD_ID).nonzero()[0].item()
        not_pad_ind = not_pad_ind - self.num_padding_at_beginning
        return torch.cat([ind_tensor[not_pad_ind:], ind_tensor[:not_pad_ind]]), \
               torch.cat([rewards[not_pad_ind:], rewards[:not_pad_ind]])

    def _transformer_forward(self, **kwargs):
        # https://github.com/microsoft/DeepSpeedExamples/issues/349
        transformer_outputs = self.rm_transformer(**kwargs)
        hidden_states = transformer_outputs[0]
        # glm模型输出的hidden_states的shape是(seq_len, bs, hidden_size),
        # 其他模型输出的是(bs, seq_len, hidden_size), 为此，这里做一个适配
        if self.is_chat_glm:
            hidden_states = hidden_states.transpose(1, 0)
        return hidden_states

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                **kwargs):
        hidden_states = self._transformer_forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            if self.score_strategy == "last":
                chosen_mean_scores.append(chosen_rewards[i][-1])
                rejected_mean_scores.append(rejected_rewards[i][-1])
                loss += -torch.log(torch.sigmoid(chosen_rewards[i] - rejected_rewards[i])).mean()
                continue

            if self.is_pad_left:
                chosen_id, chosen_reward = self._move_pad_to_end(chosen_ids[i], chosen_rewards[i])
                rejected_id, rejected_reward = self._move_pad_to_end(rejected_ids[i], rejected_rewards[i])
            else:
                chosen_id = chosen_ids[i]
                chosen_reward = chosen_rewards[i]
                rejected_id = rejected_ids[i]
                rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            # https://github.com/microsoft/DeepSpeedExamples/issues/338
            assert divergence_ind >= 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  # use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False,
                      **kwargs):
        hidden_states = self._transformer_forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            # deepspeed chat obj 无法跑
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = input_ids.shape[0]
            chosen_end_scores = []  # we use this name for consistency with the original forward function
            for i in range(bs):
                if self.score_strategy == "last":
                    chosen_end_scores.append(values[i][-1])
                    continue
                input_id = input_ids[i]
                value = values[i]
                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else 0
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }


class ProcessRewardModel(RewardModel):
    """Process Reward Model(PRM)
    REF: https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf
    """

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, extract_transformer=False,
                 score_strategy="last", head_dim=3):
        """
        Args:
            extract_transformer(bool): If True, will extract transformer out from base_model as self.rm_transformer.
                If the base_model is created by AutoModel, set extract_transformer to False.
                If the original base_model is task-specific (such as AutoModelForCausalLM), set it to True.
            score_strategy: the strategy to pick out the score from the logits of v_head. The loss is different when
                using different strategy.
                - divergence (default): use the logits of v_head whose index is the last not-pad index in input_ids.
                    the loss is calculated by the mean logits of divergent chosen- and rejected- rewards.
                - last: use the logits of v_head whose index is just the last one.
                    The loss is calculated by the mean logits of all chosen- and rejected- rewards.
                    (recommended for glm, for its higher accuracy than divergence's)
            head_dim: default to 3. PRM是一个分类模型，产出positive, negative, or neutral 3类结果，
                prm的最终score，应该为每个step的正确答案概率的乘积，但在预测的时候，会将整个seq算作一个step，且将neutral认为是
                正确的答案，所以返回positive和neutral中最大的softmax作为score。
        """
        super().__init__(base_model, tokenizer,
                         num_padding_at_beginning=num_padding_at_beginning,
                         extract_transformer=extract_transformer,
                         score_strategy=score_strategy,
                         head_dim=head_dim)
        # PRM是一个分类模型，产出positive, negative, or neutral 3类结果，而rm的最终score，应该为每个step的正确答案概率的乘积，
        # 如果只有一个，那就返回positive的概率
        self.loss = nn.CrossEntropyLoss()
        self.v_head.to(self.rm_transformer.device).to(self.rm_transformer.dtype)

    def forward(self,
                input_ids,
                targets=None,
                past_key_values=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False,
                **kwargs):
        """
        Args:
            targets: 为1D tensor, 长度为batch size, 值为非负的分类数值。注意：negative必须放在最后一列
        """
        assert targets is not None, "need set targets as labels"
        hidden_states = self._transformer_forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        values = self.v_head(hidden_states).squeeze(-1)[:, -1, :]  # 取最后一个seq，作为分类结果
        loss = self.loss(values, targets.long())
        softmax = values.softmax(dim=1)
        prediction_class = torch.argmax(softmax, dim=1)
        chosen_end_scores = softmax[:, :-1].max(dim=1).values
        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "prediction_class": prediction_class
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False,
                      **kwargs):
        hidden_states = self._transformer_forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        values = self.v_head(hidden_states).squeeze(-1)[:, -1, :]
        if return_value_only:
            return values
        else:
            softmax = values.softmax(dim=1)
            # 返回类别
            prediction_class = torch.argmax(softmax, dim=1)
            # 将neutral视为positive，两者中概率取最高。
            chosen_end_scores = softmax[:, :-1].max(dim=1).values
            return {
                "values": values,
                "chosen_end_scores": chosen_end_scores,
                "prediction_class": prediction_class
            }


def replace_key_prefix(src_dict, prefix_pairs, inplace=True):
    """Replace key from src_prefix to dest_prefix
    Args:
        src_dict: src dict.
        prefix_pairs: Pairs of [src_prefix, dest_prefix], the src_prefix will be replaced to dest_prefix.
            For example: [["transformer.", "rm_transformer.transformer."], ["v_head", "v_head"]]
        inplace: replace inplace, and the src_dict will be deleted if True.
    """
    dest_dict = {}
    for key in src_dict.keys():
        new_key = key
        for src_prefix, dest_prefix in prefix_pairs:
            if key.startswith(src_prefix):
                new_key = dest_prefix + key[len(src_prefix):]
                break
        dest_dict.update({new_key: src_dict[key]})
    if inplace:
        del src_dict
    return dest_dict


def remove_keys(src_dict, key_patterns):
    """Remove keys which match key_patterns in src_dict
    Args:
        src_dict: src dict.
        key_patterns: A list of key patterns, like ["transformer.*", ".*.attr$"]
    """
    import re

    compiled_key_regs = [re.compile(pattern) for pattern in key_patterns]
    to_del_key_set = set()
    for key in src_dict.keys():
        for reg in compiled_key_regs:
            if reg.match(key):
                to_del_key_set.update((key,))
                break
    for key in to_del_key_set:
        del src_dict[key]
    return src_dict

# This function is a modified version of code available in the from_pretrained API of HuggingFace Transformers
# The code is copied and modified from: https://github.com/huggingface/transformers/blob/5ee9693a1c77c617ebc43ef20194b6d3b674318e/src/transformers/modeling_utils.py#L498
# This function helps load a HF format checkpoint into a DeepSpeed wrapped model that has been sharded using ZeRO Stage 3
def load_rm_state_dict_from_ckpt(rm_model, model_ckpt_path, start_prefix="", zero_stage=0):
    # 不用都load，仅chief即可，
    src_state_dict = torch.load(model_ckpt_path, map_location='cpu')
    dest_prefix = "rm_transformer."
    if hasattr(rm_model.rm_transformer, "transformer"):
        dest_prefix += "transformer."
    state_dict = replace_key_prefix(src_state_dict,
                                         [["transformer.", dest_prefix], ["rwtranrsformer.", dest_prefix]])

    for extra_linear_name in ["lm_head", "qa_outputs", "score"]:
        if hasattr(rm_model.rm_transformer, extra_linear_name) and f"{extra_linear_name}.weight" not in state_dict:
            linear = getattr(rm_model.rm_transformer, extra_linear_name)
            state_dict.update({f"rm_transformer.{extra_linear_name}.weight": torch.nn.init.ones_(linear.weight)})
            if hasattr(linear, "bias"):
                state_dict.update({f"rm_transformer.{extra_linear_name}.bias": torch.nn.init.zeros_(linear.bias)})

    # 在transformers>=4.29.0版本，删除了attention的.bias 和masked_bias，所以加个处理，忽略这些keys
    import deepspeed
    import transformers
    version_fields = transformers.__version__.split(".")
    if int(version_fields[0]) > 4 or (int(version_fields[0]) == 4 and int(version_fields[1]) >= 29):
        remove_keys(state_dict, [".*.attn.bias$", ".*.attn.masked_bias$"])    

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(rm_model, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs