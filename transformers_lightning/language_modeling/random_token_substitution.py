from typing import Tuple

import torch
import transformers

from transformers_lightning.language_modeling import IGNORE_IDX, LanguageModel
from transformers_lightning.language_modeling.utils import whole_word_tails_mask


class RandomTokenSubstitution(LanguageModel):
    r"""
    Prepare tokens inputs/labels for random token substutition modeling.
    We sample a few tokens in each sequence for RTS training (with probability `probability` defaults to 0.15 in Bert/RoBERTa)
    If `whole_word_swapping` is True, either every or no token in a word will be masked. This argument requires
    that `words_tails` are passed to the `__call__` method such that the model can understand which parts of a word
    are tails ('##..'-like tokens). `words_tails` must be a boolean tensor with the same shape as `inputs`
    and be True iff the corresponding tokens starts with `##`. Passing a None `words_tails` will make the model compute
    them, which is expensive. So, for performance reasons we strongly suggest to compute `words_tails` in adapters.

    Usage example
    >>> import torch
    >>> from transformers import BertTokenizer

    >>> tok = BertTokenizer.from_pretrained("bert-base-cased")
    >>> rts = RandomTokenSubstitution(tok)

    >>> input_ids = torch.tensor([tok.encode("test sentence")])
    >>> input_ids
    ... tensor([[ 101, 2774, 5650, 102]])

    >>> swapped, labels = rts(input_ids)
    >>> swapped
    ... tensor([[ 101, 2774, 5650, 102]])
    >>> labels
    ... tensor([[-100, 1, 0, -100]]) # -100 = IGNORE_IDX

    >>> # notice that `inputs` are modified by calling `__call__`
    >>> # even if they are returned as a new output. 
    >>> input_ids is swapped
    ... True
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        probability: float = 0.15,
        whole_word_swapping: bool = False,
    ):
        super().__init__(tokenizer, probability=probability)
        self.whole_word_swapping = whole_word_swapping

    def __call__(self,
                 inputs: torch.Tensor,
                 words_tails: torch.Tensor = None) -> Tuple[torch.LongTensor, torch.LongTensor]:

        device = inputs.device
        inputs = inputs.clone()
        labels = torch.full(inputs.shape, fill_value=0, dtype=torch.long, device=device)

        # We sample a few tokens in each sequence for masked-LM training (with probability args.probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(inputs.shape, fill_value=self.probability, dtype=torch.float32, device=device)

        # create whole work masking mask -> True if the token starts with ## (following token in composed words)
        if words_tails is None and self.whole_word_swapping:
            words_tails = whole_word_tails_mask(inputs, self.tokenizer, device=device)

        if self.whole_word_swapping:
            # with whole word masking probability matrix should average probability over the entire word
            probability_matrix.masked_fill_(words_tails, value=0.0)

        # not going to substitute special tokens of the LM (bert, roby, ...)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        special_tokens_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
        probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
        labels.masked_fill_(special_tokens_mask_tensor, value=IGNORE_IDX)

        # no need to substitute padding tokens, assigning 0.0 prob
        if self.tokenizer._pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
            labels.masked_fill_(padding_mask, value=IGNORE_IDX)

        substituted_indices = torch.bernoulli(probability_matrix).bool()

        # with whole word masking, assure all tokens in a word are either all masked or not
        if self.whole_word_swapping:
            for i in range(1, substituted_indices.shape[-1]):
                substituted_indices[:, i] = substituted_indices[:, i] | (
                    substituted_indices[:, i - 1] & words_tails[:, i]
                )

        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long, device=device)
        inputs[substituted_indices] = random_words[substituted_indices]
        labels.masked_fill_(substituted_indices, value=1)

        return inputs, labels
