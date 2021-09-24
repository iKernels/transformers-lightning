from typing import Any, List

from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def whole_word_tails_mask(inputs: List[Any], tokenizer: PreTrainedTokenizerBase) -> List[Any]:
    r"""
    create whole work masking mask -> 1 if the token starts with ## (following token in composed words)
    Recursively go through the lists and convert token ids to whole word masks.
    """

    # empty list
    if not inputs:
        return inputs

    # is a list of integers, convert!
    elif isinstance(inputs[0], int):
        if isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
            return [
                token.startswith('##') for token in tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=False)
            ]
        else:
            raise ValueError(
                f"`whole_word_tails_mask` does not support {tokenizer.__class__.__name__} tokenizers."
                f"Open an issue to ask for the implementation for other tokenizer types."
            )

    # list of lists, call recursively on internal elements
    elif isinstance(inputs[0], (list, tuple)):
        return [whole_word_tails_mask(ids) for ids in inputs]

    # argument not recognized, raise error
    else:
        raise ValueError("provided incorrect input type to `_whole_word_tails_mask`")
