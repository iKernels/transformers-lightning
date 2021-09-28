from collections.abc import Iterable
from typing import Any, List

import torch
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def whole_word_tails_mask(inputs: List[Any], tokenizer: PreTrainedTokenizerBase) -> List[Any]:
    r"""
    create whole work masking mask -> 1 if the token starts with ## (following token in composed words)
    Recursively go through the lists and convert token ids to whole word masks.
    """
    if inputs is None:
        res = None

    # if is a tensor, convert to list
    is_tensor = isinstance(inputs, torch.Tensor)
    if is_tensor:
        device = inputs.device
        inputs = inputs.detach().cpu().tolist()

    # single element
    if isinstance(inputs, int):
        res = whole_word_tails_mask([inputs], tokenizer)[0]

    # is a list of integers, convert!
    elif isinstance(inputs[0], int):
        if isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
            res = [
                token.startswith('##') for token in tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=False)
            ]
        else:
            raise ValueError(
                f"`whole_word_tails_mask` does not support {tokenizer.__class__.__name__} tokenizers."
                f"Open an issue to ask for the implementation for other tokenizer types."
            )

    # list of lists, call recursively on internal elements
    elif isinstance(inputs[0], Iterable):
        res = [whole_word_tails_mask(ids, tokenizer) for ids in inputs]

    # argument not recognized, raise error
    else:
        raise ValueError("provided incorrect input type to `_whole_word_tails_mask`")

    # eventually convert back to tensor and move to original device and dtype
    if is_tensor:
        res = torch.tensor(res).to(device=device)

    return res
