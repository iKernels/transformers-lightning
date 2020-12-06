from typing import Tuple

import torch
import transformers


class LanguageModel:
    r"""
    A langugage model applies some modifications to the input sequence and eventually returns some labels.
    It is an elegant way of creating labels by starting from unlabelled text.

    The `__call__` method should be overidden and implemented by subclasses.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        raise NotImplementedError("This method must be overidden by subclasses")
