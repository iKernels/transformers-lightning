import torch
from typing import List

from transformers_lightning.language_modeling import IGNORE_IDX


def get_mini_groups(idx: torch.Tensor) -> List[torch.Tensor]:
    r"""
    Return a list of lists where each sub-list contains the indexes of each value of `idx`
    :params idx: a tensor of integer indexes

    Example:
    >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    >>> groups = get_mini_groups(indexes)
    >>> groups
    ... [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5, 6])]
    """

    indexes = dict()
    for i, _id in enumerate(idx):
        _id = _id.item()
        if _id in indexes:
            indexes[_id] += [i]
        else:
            indexes[_id] = [i]
    res = [torch.tensor(x, dtype=torch.int64) for x in indexes.values()]
    return res
