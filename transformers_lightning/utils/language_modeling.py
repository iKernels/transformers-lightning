import json
import os
import random
from typing import Union

import numpy as np
import torch
import transformers
from transformers_lightning import utils


def mask_tokens_for_mlm(
    inputs: Union[torch.Tensor, np.array],
    tokenizer: transformers.PreTrainedTokenizer,
    mlm_probability: float = 0.15,
    weights: torch.Tensor = None,
    auto_move_tensors: bool = True
):
    if isinstance(inputs, torch.Tensor):
        return _mask_tokens_for_mlm_torch(inputs,
                                          tokenizer,
                                          mlm_probability=mlm_probability,
                                          weights=weights,
                                          auto_move_tensors=auto_move_tensors)
    else:
        assert weights is None, "cannot provide weights to numpy version of mlm at the moment"
        return _mask_tokens_for_mlm_numpy(inputs,
                                          tokenizer,
                                          mlm_probability=mlm_probability)


def _mask_tokens_for_mlm_torch(
    inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    mlm_probability: float = 0.15,
    weights: torch.Tensor = None,
    auto_move_tensors: bool = True
) -> tuple:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    If weights are provided, probability of each token to be masked will be weighted.
    `inputs` and `weights` must be on the same device or `auto_move_tensors` must be true!
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer."
        )

    device = inputs.device

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    if weights is None:
        probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    else:
        if auto_move_tensors:
            weights = weights.to(device=labels.device)
        else:
            assert labels.device == weights.device, f"labels and weights must be on the same device: {labels.device} and {weights.device} found instead"
        probability_matrix = torch.softmax(weights[labels], dim=-1)
        probability_matrix = probability_matrix * probability_matrix.shape[-1] * mlm_probability

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=device), value=0.0)

    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(
        torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def _mask_tokens_for_mlm_numpy(
    input_ids: list,
    tokenizer: transformers.PreTrainedTokenizer,
    mlm_probability: float = 0.15
) -> tuple:
    """ Create a masked language model mask of a sentence. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer."
        )

    input_ids = np.array(input_ids)
    labels = input_ids.copy()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = np.full(labels.shape, mlm_probability)
    special_tokens_mask = tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)

    probability_matrix = np.ma.array(probability_matrix, mask=special_tokens_mask)
    probability_matrix = probability_matrix.filled(0.0)

    if tokenizer._pad_token is not None:
        padding_mask = (labels == tokenizer.pad_token_id)
        probability_matrix = np.ma.array(probability_matrix, mask=padding_mask)
        probability_matrix = probability_matrix.filled(0.0)

    masked_indices = np.random.binomial(1, p=probability_matrix).astype(bool)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = np.random.choice(2, p=[0.2, 0.8], size=labels.shape).astype(bool) & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = np.random.choice(2, p=[0.5, 0.5], size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
    random_words = np.random.randint(len(tokenizer), size=labels.shape).astype(int)
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, labels


def random_token_substutition(
    inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    rts_probability: float = 0.15,
    do_not_touch_input_ids=False
) -> tuple:
    """
    Prepare tokens inputs/labels for random token substutition modeling.
    We sample a few tokens in each sequence for RTS training
    (with probability args.rts_probability defaults to 0.15 in Bert/RoBERTa)
    
    Example
    >>> from transformers import BertTokenizer
    >>> tok = BertTokenizer.from_pretrained("bert-base-cased")

    >>> bs = tok.encode_plus("ciao sono luca")
    >>> bs = {k: torch.tensor(v).unsqueeze(0) for k,v in bs.items()}
    >>> print(bs)
    {
        'input_ids': tensor([[  101,   172,  1465,  1186,  1488,  1186,   181, 23315,   102]]),
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
    }

    >>> bs["input_ids"], bs["labels"] = random_token_substutition(
    >>>     bs["input_ids"], tok, rts_probability=0.4
    >>> )
    >>> print(bs)
    {
        'input_ids': tensor([[  101,   172,  8959,  1186,  3082, 27735,   181,  1348,   102]]),
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        'labels': tensor([[-100, 0, 1, 0, 1, 1, 0, 1, -100]])
    }

    If `do_not_touch_input_ids` is True, input ids are not modified and only labels are created
    """

    # creating tensor directly on right device is more efficient
    device = inputs.device

    # probability matrix containts the probability of each token to be randomly substituted
    probability_matrix = torch.full(inputs.shape, fill_value=rts_probability, dtype=torch.float32, device=device)
    labels = torch.full(inputs.shape, fill_value=0, dtype=torch.long, device=device)

    # not going to substitute special tokens of the LM (bert, roby, ...)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
    labels.masked_fill_(special_tokens_mask_tensor, value=-100)

    # no need to substitute padding tokens, assigning 0.0 prob
    if tokenizer._pad_token is not None:
        padding_mask = inputs.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        labels.masked_fill_(padding_mask, value=-100)

    substituted_indices = torch.bernoulli(probability_matrix).bool()

    random_words = torch.randint(len(tokenizer), inputs.shape, dtype=torch.long, device=device)
    if not do_not_touch_input_ids:
        inputs[substituted_indices] = random_words[substituted_indices]
    labels.masked_fill_(substituted_indices, value=1)

    return inputs, labels
