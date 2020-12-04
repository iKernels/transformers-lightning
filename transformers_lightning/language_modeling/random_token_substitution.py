from typing import Tuple
import torch

import transformers
from pytorch_lightning import _logger as logger
import scipy.stats as st

from transformers_lightning import utils
from transformers_lightning.language_modeling import IGNORE_IDX, LanguageModel
from transformers_lightning.language_modeling.utils import whole_word_lists


class RandomTokenSubstitution(LanguageModel):
    """
    Prepare tokens inputs/labels for random token substutition modeling.
    We sample a few tokens in each sequence for RTS training (with probability `rts_probability` defaults to 0.15 in Bert/RoBERTa)

    If `weights` are provided, probability of each token to be masked will be weighted in the following way:
        - w are the weights
        - p the probability of masking a token

        probabilities = ( 1 + (w - w_mean) / (w_std * Z_(1 - r/2)) ) -> clipped in [0, 1]

    If `whole_word_swapping` is True, either every or no token in a word will be masked.

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
        rts_probability: float = 0.15,
        reliability: float = 0.05,
        whole_word_swapping: bool = False,
    ):
        super().__init__(tokenizer)
        self.rts_probability = rts_probability
        self.reliability = reliability
        self.whole_word_swapping = whole_word_swapping

    def __call__(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:

        device = inputs.device
        labels = torch.full(inputs.shape, fill_value=0, dtype=torch.long, device=device)

        # We sample a few tokens in each sequence for masked-LM training (with probability args.rts_probability defaults to 0.15 in Bert/RoBERTa)
        if weights is None:
            probability_matrix = torch.full(inputs.shape, fill_value=self.rts_probability, dtype=torch.float32, device=device)
        else:
            weights = weights.to(device=device)
            z_score = st.norm.ppf(1 - self.reliability / 2)
            probability_matrix = self.rts_probability * (1 + utils.normalize_standard(weights[inputs], dim=-1) / z_score)
            probability_matrix = torch.clip(probability_matrix, min=0, max=1)
        
        # create whole work masking mask -> 1 if the token starts with ## (following token in composed words)
        whole_words_array = whole_word_lists(inputs, self.tokenizer) if self.whole_word_swapping else None

        if self.whole_word_swapping:
            # with whole word masking probability matrix should average probability over the entire word
            for i, whole_words in enumerate(whole_words_array):
                for word in whole_words:
                    if len(word) > 1:
                        probability_matrix[i, word[0]] = probability_matrix[i, word].mean()
                        probability_matrix[i, word[1:]] = 0

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

        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long, device=device)
        inputs[substituted_indices] = random_words[substituted_indices]
        labels.masked_fill_(substituted_indices, value=1)

        # with whole word masking, assure every token in a word is either masked in each token or not
        if self.whole_word_swapping:
            tail_indices = torch.zeros_like(substituted_indices).bool()
            for i, whole_words in enumerate(whole_words_array):
                for word in whole_words:
                    if labels[i, word[0]]:
                        tail_indices[i, word[1:]] = True

            labels[tail_indices] = True
            inputs[tail_indices] = random_words[tail_indices]

        return inputs, labels
