import torch
from transformers import PreTrainedTokenizer


def whole_word_tails_mask(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, device: torch.device = None
) -> torch.BoolTensor:
    r"""
   # create whole work masking mask -> 1 if the token starts with ## (following token in composed words)
   """
    return torch.tensor(
        [
            [token.startswith('##')
             for token in tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)]
            for ids in inputs
        ],
        device=device,
        dtype=torch.bool
    )
