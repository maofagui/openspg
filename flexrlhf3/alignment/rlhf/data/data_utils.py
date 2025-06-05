# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import random
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
from itertools import chain
from alignment.rlhf.data import raw_datasets


def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    if dataset_name == "Dahoas/rm-static":
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank)
    elif dataset_name == "Dahoas/full-hh-rlhf":
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank)
    elif dataset_name == "Dahoas/synthetic-instruct-gptj-pairwise":
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank)
    elif dataset_name == "yitingxie/rlhf-reward-datasets":
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank)
    elif dataset_name == "openai/webgpt_comparisons":
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank)
    elif dataset_name == "stanfordnlp/SHP":
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank)
    elif dataset_name == "wangrui6/Zhihu-KOL":
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank)
    elif dataset_name == "Cohere/miracl-zh-queries-22-12":
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank)
    elif dataset_name == "Hello-SimpleAI/HC3-Chinese":
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank)
    elif dataset_name == "mkqa-Chinese":
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank)
    elif dataset_name == "mkqa-Japanese":
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank)
    elif dataset_name == "Cohere/miracl-ja-queries-22-12":
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank)
    elif dataset_name == "lmqg/qg_jaquad":
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank)
    elif dataset_name == "lmqg/qag_jaquad":
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_range, split_index,
                                data_size):
    if isinstance(data_range, str):
        data_split = data_range
        index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
        if not os.path.isfile(index_file_name) and local_rank <= 0:
            splits = [float(s) for s in data_split.split(',')]
            splits_sum = sum(splits)
            splits = [split / splits_sum for split in splits]
            splits_index = [0]
            for index, split in enumerate(splits):
                splits_index.append(splits_index[index] +
                                    int(round(split * float(data_size))))
            diff = splits_index[-1] - data_size
            for index in range(1, len(splits_index)):
                splits_index[index] -= diff
            assert splits_index[-1] == data_size

            shuffle_idx = get_shuffle_idx(seed, data_size)
            for split_i in range(len(splits)):
                shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
                shuffle_idx_split = shuffle_idx[splits_index[split_i]:splits_index[split_i + 1]]
                np.save(shuffle_idx_split_file_name,
                        shuffle_idx_split,
                        allow_pickle=True)
        torch.distributed.barrier()
        index = np.load(index_file_name, allow_pickle=True)
        return index.tolist()
    else:
        data_split = "_".join([str(i) for i in data_range])
        index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}.npy"
        if not os.path.isfile(index_file_name) and local_rank <= 0:
            shuffle_idx = get_shuffle_idx(seed, data_size)
            splits_index = [int(round(data_range[0] * data_size)), int(round(data_range[1] * data_size))]
            shuffle_idx_split = shuffle_idx[splits_index[0]:splits_index[1]]
            np.save(index_file_name, shuffle_idx_split, allow_pickle=True)
        torch.distributed.barrier()
        index = np.load(index_file_name, allow_pickle=True)
        return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                   self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], \
                   self.pad_token_id


def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)

    elif train_phase == 3:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                for key_word in ["input_ids", "attention_mask"]:
                    length = prompt_token[key_word].size()[-1]
                    if length > max_seq_len:
                        y = prompt_token[key_word].squeeze(0)[length -
                                                              (max_seq_len -
                                                               1):].flip(0)
                    else:
                        y = prompt_token[key_word].squeeze(0).flip(0)
                    prompt_token[key_word] = y
                prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_range,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>"):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = '_'.join(data_path)
    tokenizer_name = tokenizer.init_kwargs['name_or_path'].replace('/', '_')
    split_by_range = not isinstance(data_range, str)
    data_split = "_".join([str(i) for i in data_range]) if split_by_range else data_range
    fname = f"{fname}_split{data_split}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}"
    fname = '_'.join(fname.split('/'))
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    if local_rank >= 0:
        torch.distributed.all_reduce(buf_create_cache)

    # Skip creating cache if we found it on all the nodes.
    if buf_create_cache.item() == 0:
        return torch.load(train_fname), torch.load(eval_fname)
    else:
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank, data_path[0], data_range, output_path, train_phase,
                seed, tokenizer, end_of_conversation_token, max_seq_len)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank, data_path[0], data_range, output_path, train_phase,
                    seed, tokenizer, end_of_conversation_token, max_seq_len)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        if local_rank <= 0:
            torch.save(train_dataset, train_fname)
            torch.save(eval_dataset, eval_fname)
        return train_dataset, eval_dataset


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size=None):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        """
        data is like:
        """
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a sequence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(pad_length, 0),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(pad_length, 0),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)  # 按照维度翻转
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


def get_unsupervised_data(context, tokenizer):
    data_config = context.train_conf.data_conf
    unsupervised_raw_datasets = load_dataset(
        data_config.unsupervised_dataset_name, data_config.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_config.num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = context.model_conf.max_prompt_seq_len + context.model_conf.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
                [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_config.num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def separate(self, dataset):
        """ 将self.dataset中的大batch数据，按照small_batch_size进行均匀分割，
        small_dataset是一个len=max_size/small_batch_size的数组
        """
        small_dataset = []
        for large_batch in dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)

            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size] if v is not None else None
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i + self.small_batch_size])

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                sep = self.separate(self.dataset)
                self.free()
                return sep
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []

    def has_remaining(self):
        return len(self.dataset) > 0


class RolloutStorage(MiniDataset):

    def __init__(self, sample_size, small_batch_size, queue_size=None):
        """实现一个循环队列
        Args:
            sample_size(int): 一次sample的数据量
            small_batch_size(int): sample时用于拆分队列数据的batch_size
            queue_size(int): 队列大小，当队列满了的时候，才开始sample。新增的数据按照FIFO覆盖。
        """
        super().__init__(sample_size, small_batch_size)
        self.pos = 0
        self.queue_size = queue_size or sample_size * 8
        self.set_sample_method()

    def add(self, data):
        if len(self.dataset) >= self.queue_size:
            self.dataset[self.pos % self.queue_size] = data
            self.pos += 1
        else:
            self.dataset.append(data)
        return self._sample() if len(self.dataset) >= self.queue_size else None

    def set_sample_method(self, method='uniform'):
        self.method = method

    def _sample(self):
        if self.method == 'uniform':
            indices = np.random.choice(self.queue_size, self.max_size, replace=False)
            samples = [self.dataset[i] for i in indices]
            return self.separate(samples)
        else:
            raise NotImplementedError()


class BasePromptDataset(Dataset):

    def __init__(self, prompt_dataset, pad_token_id) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.prompt_dataset)

    def __getitem__(self, idx):
        data_item = self.prompt_dataset[idx]
        return data_item["input_ids"], data_item["attention_mask"] if "attention_mask" in data_item else None, \
               self.pad_token_id


class CustomDataCollator:

    def __init__(self, max_token_len, inference_tp_size=None, padding_side="left"):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size
        self.padding_side = padding_side

    def __call__(self, data):
        """这里输入的data，就是Dataset(如SimplePromptDataset)里__getitem__的返回。
        """
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final output is a sequence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(pad_length, 0),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(pad_length, 0),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        # 按照维度翻转
        if self.padding_side == "left":
            batch["prompt"] = batch["prompt"].flip(1)
            batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


class TokenizeDataCollator:

    def __init__(self, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, data):
        """data为一个batch的prompt"""
        batch = {}
        prompt_token = self.tokenizer(data,
                                      truncation=True,
                                      max_length=self.max_token_len,
                                      padding="max_length",
                                      return_tensors="pt")
        batch["prompt"] = prompt_token["input_ids"]
        batch["prompt_att_mask"] = prompt_token["attention_mask"]
        return batch


def build_prompt_dataset(current_dataset, tokenizer, max_seq_len):
    """ 将prompt经过tokenizer进行编码，并封装成Dataset，用户需要根据实际情况调整。"""
    prompt_dataset = []
    padding_side = tokenizer.padding_side.lower() if hasattr(tokenizer, "padding_side") else "left"
    for i, prompt in enumerate(current_dataset):
        # tokenize the prompt (current_dataset = train_split_ds['prompt'])
        if prompt is not None:
            prompt_token = tokenizer(prompt, return_tensors="pt")
            for key_word in ["input_ids", "attention_mask"]:
                if key_word not in prompt_token:
                    continue
                length = prompt_token[key_word].size()[-1]
                if length > max_seq_len:
                    y = prompt_token[key_word].squeeze(0)[length - (max_seq_len - 1):]
                else:
                    y = prompt_token[key_word].squeeze(0)
                # 如果padding_side为left，则对token进行翻转，并在data_collate里，在进行pad之后翻转恢复
                prompt_token[key_word] = y.flip(0) if padding_side == "left" else y
            prompt_dataset.append(prompt_token)
    return BasePromptDataset(prompt_dataset, tokenizer.pad_token_id)


def instantiate_dataloader(split_dataset, tokenizer, batch_size, max_seq_len, sampler_cls=None, shuffle=False):
    from alignment.api.utils import dist_util
    from torch.utils.data import RandomSampler, DistributedSampler, DataLoader

    train_ds = build_prompt_dataset(split_dataset, tokenizer, max_seq_len)
    data_collator = CustomDataCollator(max_seq_len, padding_side=(
        tokenizer.padding_side.lower() if hasattr(tokenizer, "padding_side") else "left"))
    local_rank = dist_util.get_local_rank()
    if sampler_cls:
        ds_sampler = sampler_cls(train_ds)
    elif local_rank == -1:
        ds_sampler = RandomSampler(train_ds)
    else:
        ds_sampler = DistributedSampler(train_ds)
    ds_dataloader = DataLoader(
        train_ds,
        shuffle=shuffle,
        collate_fn=data_collator,
        sampler=ds_sampler,
        batch_size=batch_size)
    return ds_dataloader


def create_dataloader(iter_dataset: IterableDataset, tokenizer, batch_size, max_seq_len):
    """数据处理流程：
    odps table->[OdpsIterableDataset(->set_fn)->transform] -> prompts ->
    [CustomDataCollator(->init)->batch->call] -> tokenized batch
    其中tokenized batch的格式为{ "prompt": [[input_ids], ...], "prompt_att_mask": [[attention_mask], ...]}

    Args:
        iter_dataset: An instance of IterableDataset.
        tokenizer: The tokenizer is used in collate_fn.
        batch_size: batch size.
        max_seq_len: Used when tokenizing prompt.
    """
    ds_dataloader = DataLoader(
        iter_dataset,
        collate_fn=TokenizeDataCollator(tokenizer, max_seq_len),
        sampler=None,
        batch_size=batch_size)

    return ds_dataloader


def wrap_to_dataloader_fn(*dataloader_args, **dataloader_kwargs):
    """将实例化DataLoader的函数，及其参数分开返回，用于延迟调用"""
    import functools

    dataloader_fn = functools.partial(DataLoader, *dataloader_args, **dataloader_kwargs)
    return dataloader_fn, ()


class SEPMiniDataset:
    def __init__(self, batch_size, micro_batch_size):
        self.dataset = []
        self.cur_dataset = []
        self.max_size = batch_size
        self.small_batch_size = micro_batch_size
        self.num_micro_batches = int(batch_size // micro_batch_size)
        self._epoch = 0
        self._cur_index = 0

    def separate(self, dataset):
        """ 将self.dataset中的大batch数据，按照small_batch_size进行均匀分割，
        small_dataset是一个len=max_size/small_batch_size的数组

        注意后续改成补齐batch个数，不然会卡
        """
        small_dataset = []
        for large_batch in dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)

            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append([x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append(
                        {k: v[i:i + self.small_batch_size] if v is not None else None
                         for k, v in large_batch.items()})
                else:
                    small_dataset.append(large_batch[i:i + self.small_batch_size])

        return small_dataset

    def add(self, data):
        cur_dataset = self.separate(data)
        self.cur_dataset.extend(cur_dataset)
        self.dataset.extend(cur_dataset)

    def next_batch(self):
        self._cur_index += 1
        if self._epoch > 0:
            return self.dataset[(self._cur_index - 1) * self.num_micro_batches:self._cur_index *
                                self.num_micro_batches]
        else:
            if len(self.cur_dataset) == self.num_micro_batches:
                ret = self.cur_dataset
                self.cur_dataset = []
                return ret

    def free(self):
        self.dataset = []

    def set_epoch(self, epoch):
        self._epoch = epoch
        self._cur_index = 0
        if epoch > 0:
            random.shuffle(self.dataset)