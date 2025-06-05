from alignment.rlhf.data.data_utils import create_prompt_dataset, get_unsupervised_data, DataCollatorRLHF
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader

from transformers import default_data_collator


def create_datasets(context, tokenizer, train_phase, data_conf):
    unsupervised_training_enabled = data_conf.unsupervised_dataset_name and data_conf.unsupervised_dataset_config_name
    prompt_train_dataset, _ = create_prompt_dataset(
        context.local_rank, data_conf.dataset, data_conf.data_range,
        data_conf.workdir, train_phase, context.runtime_conf.seed, tokenizer,
        context.model_conf.max_prompt_seq_len)
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(context, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    # data_collator = DataCollatorRLHF(args.max_prompt_seq_len, args.inference_tp_size)
    data_collator = DataCollatorRLHF(context.model_conf.max_prompt_seq_len)
    if context.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=data_conf.batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=data_conf.batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    return prompt_train_dataloader, unsupervised_train_dataloader
