from typing import List
import torch

from transformers import PreTrainedTokenizer


def whole_word_lists(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer) -> List[List[int]]:
    """
    # create whole work masking mask -> 1 if the token starts with ## (following token in composed words)
    """
    res = []
    for ids in inputs:
        base = []
        for i, token in enumerate(tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)):
            if not token.startswith('##'): base.append([i])
            else: base[-1].append(i)
        res.append(base)
    return res

