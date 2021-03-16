from typing import Tuple

import torch
import transformers


class LanguageModel:
    r"""
    A langugage model applies some modifications to the input sequence and eventually returns some labels.
    It is an elegant way of creating labels by starting from unlabelled text.

    Args:
        `tokenizer`: a huggingface-compatible tokenizer to extract pad token id, mask token id and other ids
        `probability`: probability of a token being modified. It can be interpreted in different ways by
            subclasses

    The `__call__` method should be overidden and implemented by subclasses.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, probability: float = 0.15):
        self.tokenizer = tokenizer

        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Argument `probability` must be a float between 0.0 and 1.0, found: {probability}")
        self.probability = probability

    def __call__(self, input_ids: torch.LongTensor, *args, **kwargs) -> Tuple:
        raise NotImplementedError("This method must be overidden by subclasses")
