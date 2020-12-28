import numpy as np
import torch
import transformers
import scipy.stats as st
from transformers_lightning import utils
from pytorch_lightning import _logger as logger

IGNORE_IDX = -100


def mask_tokens_for_mlm(
    inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    mlm_probability: float = 0.15,
    weights: torch.Tensor = None,
    reliability: float = 0.05,
    **kwargs
) -> tuple:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    If weights are provided, probability of each token to be masked will be weighted.
    `inputs` and `weights` must be on the same device or `auto_move_tensors` must be true!
    """

    if 'auto_move_tensors' in kwargs:
        logger.warning("`auto_move_tensors` is deprecated and will be removed in 0.3.0")

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer."
        )

    device = inputs.device

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    if weights is None:
        probability_matrix = torch.full(labels.shape, fill_value=mlm_probability, dtype=torch.float32, device=device)
    else:
        weights = weights.to(device=labels.device)

        """
        Going to use the weights in the following way:
        - w are the weights
        - p the probability of masking a token

        probabilities = ( 1 + (w - w_mean) / (w_std * Z_(1 - r/2)) ) -> clipped in [0, 1]
        """
        z_score = st.norm.ppf(1 - reliability/2)
        probability_matrix = mlm_probability * (1 + utils.normalize_standard(weights[labels], dim=-1) / z_score)
        probability_matrix = torch.clip(probability_matrix, min=0, max=1)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=device), value=0.0)

    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = IGNORE_IDX  # We only compute loss on masked tokens

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


def random_token_substutition(
    inputs: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    rts_probability: float = 0.15,
    do_not_touch_input_ids=False,
    weights: torch.Tensor = None,
    reliability: float = 0.05,
    ngrams:int = 1,
    **kwargs
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
        'labels': tensor([[IGNORE_IDX, 0, 1, 0, 1, 1, 0, 1, IGNORE_IDX]])
    }

    If `do_not_touch_input_ids` is True, input ids are not modified and only labels are created.
    If `weights` if not None, the probability matrix will be initialized giving each token a probability
    proportional to the weight. `weights` should be a vector on length `vocab_size`.
    """
    if 'auto_move_tensors' in kwargs:
        logger.warning("`auto_move_tensors` is deprecated and will be removed in 0.3.0")

    # creating tensor directly on right device is more efficient
    device = inputs.device

    # We sample a few tokens in each sequence for masked-LM training (with probability args.rts_probability defaults to 0.15 in Bert/RoBERTa)
    if weights is None:
        probability_matrix = torch.full(inputs.shape, fill_value=rts_probability, dtype=torch.float32, device=device)
    else:
        weights = weights.to(device=device)

        """
        Going to use the weights in the following way:
        - w are the weights
        - p the probability of masking a token

        probabilities = ( 1 + (w - w_mean) / (w_std * Z_(1 - r/2)) ) -> clipped in [0, 1]
        """
        z_score = st.norm.ppf(1 - reliability/2)
        probability_matrix = rts_probability * (1 + utils.normalize_standard(weights[inputs], dim=-1) / z_score)
        probability_matrix = torch.clip(probability_matrix, min=0, max=1)

    # probability matrix containts the probability of each token to be randomly substituted
    labels = torch.full(inputs.shape, fill_value=0, dtype=torch.long, device=device)

    # not going to substitute special tokens of the LM (bert, roby, ...)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask_tensor = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
    labels.masked_fill_(special_tokens_mask_tensor, value=IGNORE_IDX)

    # no need to substitute padding tokens, assigning 0.0 prob
    if tokenizer._pad_token is not None:
        padding_mask = inputs.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        labels.masked_fill_(padding_mask, value=IGNORE_IDX)

    substituted_indices = torch.bernoulli(probability_matrix).bool()

    random_words = torch.randint(len(tokenizer), inputs.shape, dtype=torch.long, device=device)
    if not do_not_touch_input_ids:
        inputs[substituted_indices] = random_words[substituted_indices]
    labels.masked_fill_(substituted_indices, value=1)

    # compute the ngrams labels
    if ngrams > 1:
        labels = make_ngrams_labels(labels, ngrams, ignore_idx=IGNORE_IDX)

    return inputs, labels


def make_ngrams_labels(labels, n, ignore_idx=None):

    if ignore_idx is None:
        grams = labels.repeat(1, n).view(labels.size(0), n, -1)
    else:
        grams = labels[:,1:].repeat(1, n).view(labels.size(0), n, -1)

    for i in range(1, grams.size(1)):        
        grams[:,i] |= torch.roll(grams[:,i-1], 1)
        
        if ignore_idx is not None:
            grams[:,i][grams[:,i] < 0] = ignore_idx

    for i in range(1, grams.size(1)):                     
        grams[:,i,:i] = ignore_idx 
        
    if ignore_idx is not None:
        v = (torch.zeros(grams.size(0) * n, dtype=torch.long) + ignore_idx).view(grams.size(0), n, 1).to(labels.device)
        grams = torch.cat([v, grams], dim=2)

    return grams.permute(0,2,1).contiguous()
