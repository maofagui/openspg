# coding: utf-8

import torch

from alignment.app.util import logger


class MixedModel():
    def __init__(self, place_policy, init_model_fn=None, config=None):
        self._place_policy = place_policy
        self._init_model_fn = init_model_fn
        self._model = None
        self._config = config

    @property
    def model(self):
        if self._place_policy.is_owner:
            if self._model is None:
                self._model = self._init_model_fn(self._place_policy.device_id).to(f'cuda:{self._place_policy.group_owner_id}')
                logger.info(f'Put model: {self} to {self._place_policy.group_owner_id}')
            return self._model

    def _gather_device_data(self, data):
        if not torch.distributed.is_initialized():
            return [data]
        # TODO: None数据的处理
        if self._place_policy.is_owner:
            gathered_data = [
                torch.empty(data.shape, dtype=data.dtype, device=data.device)
                for _ in range(self._place_policy.group_size)
            ]
            torch.distributed.gather(data,
                                     gathered_data,
                                     dst=self._place_policy.group_owner_id,
                                     group=self._place_policy.dist_groups)
            return gathered_data
        else:
            torch.distributed.gather(data, dst=self._place_policy.group_owner_id, group=self._place_policy.dist_groups)

    def _scatter_device_data(self, all_data, shape, device=None):
        if not torch.distributed.is_initialized():
            return data[0]

        # TODO: None数据的处理
        data = torch.empty(shape, device=f'cuda:{self._place_policy.device_id}')
        torch.distributed.scatter(data, all_data, group=self._place_policy.dist_groups, src=self._place_policy.group_owner_id)
        return data


class RewardModel(MixedModel):
    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):
        # logger.info(f'{locals()}')

        # TODO: 合并数据
        gathered_input_ids = self._gather_device_data(input_ids)
        gathered_attention_mask = self._gather_device_data(attention_mask)
        all_values = []
        all_chosen_values = []
        if self._place_policy.is_owner:
            # logger.info(f'gathered_input_ids {gathered_input_ids} gathered_attention_mask {gathered_attention_mask}')
            for _input_ids, _attention_mask in zip(gathered_input_ids, gathered_attention_mask):
                origin_res = self.model.forward_value(_input_ids, _attention_mask, None, None, None, None,
                                                      return_value_only, prompt_length, use_cache)
                all_values.append(origin_res['values'])
                all_chosen_values.append(origin_res['chosen_end_scores'])

        print(all_values, all_chosen_values)
        
        
        values = self._scatter_device_data(all_values, input_ids.shape)
        chosen_values = self._scatter_device_data(all_chosen_values, (input_ids.shape[0], ))

        # 交给reward
        return {'values': values, 'chosen_end_scores': chosen_values}

    def eval(self):
        if self.model is not None:
            self.model.eval()


class RefModel(MixedModel):
    def __call__(self, input_ids=None, attention_mask=None):
        from transformers.modeling_outputs import CausalLMOutputWithPast
        # logger.info(f'{locals()}')

        # TODO: 合并两个入参
        gathered_input_ids = self._gather_device_data(input_ids)
        gathered_attention_mask = self._gather_device_data(attention_mask)
        all_logits = []
        if self._place_policy.is_owner:
            # logger.info(f'gathered_input_ids {gathered_input_ids} gathered_attention_mask {gathered_attention_mask}')
            for _input_ids, _attention_mask in zip(gathered_input_ids, gathered_attention_mask):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    origin_res = self.model(_input_ids, _attention_mask)
                all_logits.append(origin_res.logits)

        # print(all_logits)

        logit_shape = input_ids.shape + (self._config.vocab_size, )
        logits = self._scatter_device_data(all_logits, logit_shape)

        return CausalLMOutputWithPast(logits=logits)

    def eval(self):
        if self.model is not None:
            self.model.eval()
