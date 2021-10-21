from typing import Tuple

import torch
import transformers

from transformers_lightning.language_modeling import IGNORE_IDX, LanguageModel
from transformers_lightning.language_modeling.utils import create_position_ids_from_input_ids


class SwappedLanguageModeling(LanguageModel):
    r"""
    Prepare tokens inputs/labels for swapped language modeling.
    We sample a few tokens in each sequence for swapped language modeling (with probability `probability`).

    In this Language Modeling, tokens are not touched. What is modified is their position, that is swapped between
    different tokens and re-predicted.

    Example:
        >>> import torch
        >>> from transformers import BertTokenizer

        >>> tok = BertTokenizer.from_pretrained("bert-base-cased")
        >>> sort = SwappedLanguageModeling(tok)

        >>> input_ids = torch.tensor([tok.encode("what can transformers do?")])
        >>> # tokens: ['[CLS]', 'what', 'can', 'transform', '##ers', 'do', '?', '[SEP]']

        >>> input_ids
        tensor([[101, 1184, 1169, 11303, 1468, 1202, 136, 102]])

        >>> position_mask, position_labels = sort(input_ids)
        >>> position_mask
        tensor([[0, 2, 1, 3, 4, 7, 6, 5, 8]])
        >>> position_labels
        tensor([[-100, 1, 2, -100, -100, 5, -100, 7]]) # -100 = IGNORE_IDX
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        probability: float = 0.15,
    ):
        super().__init__(tokenizer, probability=probability)

    def __call__(self,
                 inputs: torch.Tensor,
                 past_key_values_length: int = 0) -> Tuple[torch.LongTensor, torch.LongTensor]:

        device = inputs.device
        batch_size, max_sequence_length = inputs.shape

        # We sample a few tokens in each sequence for sorting language modeling training
        # (with probability self.probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(inputs.shape, fill_value=self.probability, dtype=torch.float32, device=device)

        # not going to substitute special tokens of the LM (bert, roby, ...)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        special_tokens_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
        probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)

        # no need to substitute padding tokens, assigning 0.0 prob
        if self.tokenizer._pad_token is not None:
            padding_mask = (inputs == self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        position_ids = create_position_ids_from_input_ids(
            input_ids=inputs, padding_idx=self.tokenizer.pad_token_id, past_key_values_length=past_key_values_length
        )
        position_mask = torch.bernoulli(probability_matrix).bool()
        position_labels = position_ids.masked_fill(~position_mask, value=IGNORE_IDX)

        # slightly faster than moving everything to gpu
        relevant_indexes_number = position_mask.sum(dim=-1).to(device=torch.device('cpu'))
        cumsum = torch.cumsum(torch.cat([torch.tensor([0]), relevant_indexes_number[:-1]]), dim=0)
        indexes = torch.cat([torch.randperm(rel) + cums
                             for rel, cums in zip(relevant_indexes_number, cumsum)]).to(device=device)
        position_ids[position_mask] = position_ids[position_mask][indexes]

        return position_ids, position_labels
