# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union
import random

import torch
import numpy as np

import nltk
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.debug_utils import DebugOption
from transformers.utils import logging

logger = logging.get_logger(__name__)

SEED = 5566
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

class Seq2SeqTrainer(Trainer):
    def __init__(
        self, *args, top_k=None, top_p = None, 
        temperature = None, do_sample=None, 
        max_length=None, num_beams=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_length = max_length 

    # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/trainer.py#L209
    # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/trainer_seq2seq.py#L31
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # self._memory_tracker.start()

        # eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # if(self.args.use_legacy_prediction_loop):
        #     print("using self.prediction_loop")
        # else:
        #     print("using self.evaluation_loop")
        # eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        # output = eval_loop(
        #     eval_dataloader,
        #     description="Evaluation",
        #     # No point gathering the predictions if there are no metrics, otherwise we defer to
        #     # self.args.prediction_loss_only
        #     prediction_loss_only= None,
        #     ignore_keys=ignore_keys,
        #     metric_key_prefix=metric_key_prefix,
        # )

        # self.log(output.metrics)

        # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        # self._memory_tracker.stop_and_update_metrics(output.metrics)

        # return output.metrics
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
        <Tip>
        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # gen_kwargs = self._gen_kwargs.copy()
        # if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
        #     gen_kwargs["max_length"] = self.model.config.max_length
        # gen_kwargs["num_beams"] = (
        #     gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        # )
        # default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        # gen_kwargs["synced_gpus"] = (
        #     gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        # )
        gen_kwargs = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature" : self.temperature,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        # elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
        #     gen_kwargs["max_new_tokens"] + 1
        # ):
        #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)


        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        from tw_rouge import get_rouge
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
            labels_copy = labels.clone()

        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss