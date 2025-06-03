# coding: utf-8

import torch
from typing import Dict, Union, Any, Optional, List
from alignment.util.global_vars import global_context
from alignment.rlhf.utils.utils import nested_detach


class SFTTrainModule:

    def __init__(self, engine, tokenizer, context=None):
        self.sft_engine = engine
        self.tokenizer = tokenizer
        self.context = context or global_context()

        self.model = self.sft_engine.model

    def train(self):
        """set train mode"""
        self.model.train()

    def eval(self):
        """set eval mode"""
        self.model.eval()

    def do_train(self, inputs, **kwargs):
        outputs = self.model(**inputs, use_cache=False)
        assert hasattr(outputs, "loss"), \
            "missing loss in outputs, may be caused by " \
            "1)no labels in dataloader, 2)no loss is returned by the forward() of the model class"
        loss = self.compute_loss(self.model, inputs, return_outputs=False)
        # loss.requires_grad_(True)
        self.model.backward(loss)
        self.model.step()
        return loss

    def prediction_step(self,
                        inputs: Dict[str, Union[torch.Tensor, Any]],
                        prediction_loss_only: bool = True,
                        ignore_keys: Optional[List[str]] = None, ):
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits (each being optional).
        """
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)
            loss = loss.detach()

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]

        if prediction_loss_only:
            return (loss, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
