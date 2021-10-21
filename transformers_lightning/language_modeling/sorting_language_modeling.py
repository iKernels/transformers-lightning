from typing import Tuple

import torch
import transformers

from transformers_lightning.language_modeling import IGNORE_IDX, LanguageModel
from transformers_lightning.language_modeling.utils import create_position_ids_from_input_ids


class SortingLanguageModeling(LanguageModel):
    r"""
    Prepare tokens inputs/labels for sorting language modeling.
    We sample a few tokens in each sequence for sorting language modeling (with probability `probability`).

    In this Language Modeling, tokens are not touched. What is modified is their position, that is masked and
    put into labels to be predicted.

    Example:
        >>> import torch
        >>> from transformers import BertTokenizer

        >>> tok = BertTokenizer.from_pretrained("bert-base-cased")
        >>> sort = SortingLanguageModeling(tok)

        >>> input_ids = torch.tensor([tok.encode("what can transformers do?")])
        >>> # tokens: ['[CLS]', 'what', 'can', 'transform', '##ers', 'do', '?', '[SEP]']

        >>> input_ids
        tensor([[101, 1184, 1169, 11303, 1468, 1202, 136, 102]])

        >>> position_ids, position_labels = sort(input_ids)
        >>> position_ids
        tensor([[0, 1, 10, 3, 4, 10, 6, 7, 8]]) # 10: mask like position id
        >>> position_labels
        tensor([[-100, -100, 2, -100, -100, 5, -100, -100]]) # -100 = IGNORE_IDX
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        hide_position_id: int,
        probability: float = 0.15,
    ):
        super().__init__(tokenizer, probability=probability)

        if hide_position_id is None:
            raise ValueError("You must define a positive integer `hide_position_id`")
        self.hide_position_id = hide_position_id

    def __call__(
        self,
        inputs: torch.Tensor,
        past_key_values_length: int = 0
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:

        device = inputs.device

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
        position_ids.masked_fill_(position_mask, value=self.hide_position_id)

        return position_ids, position_labels
