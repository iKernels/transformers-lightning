import torch


# TODO: adapt to pytorch_lightning template
def get_mini_groups(idx: torch.Tensor) -> list:
    """
    Return a list of lists where each sub-list contains the indexes of some group of `idx`
    [0, 0, 0, 1, 1, 1, 1] -> [[0, 1, 2], [3, 4, 5, 6]]
    """
    indexes = dict()
    for i, _id in enumerate(idx):
        _id = _id.item()
        if _id in indexes:
            indexes[_id] += [i]
        else:
            indexes[_id] = [i]
    return indexes.values()

def normalize(x: torch.Tensor, do_softmax: bool = False, do_argmax: bool = False) -> torch.Tensor:
    # softmax and argmax to obtain discrete predictions probabilities
    if do_softmax:
        x = torch.nn.functional.softmax(x, dim=-1)
    if do_argmax:
        x = torch.argmax(x, dim=-1)
    return x

def reciprocal_rank(preds: torch.Tensor, labels: torch.Tensor):
    """
    RR over a single group. See `get_mini_groups` for details about groups
    """
    labels = labels[torch.argsort(preds, dim=-1, descending=True)]
    position = torch.where(labels == 1)[0]
    return 1.0 / (position[0] + 1) if (len(position.shape) > 0) and (position.shape[0] > 0) else torch.tensor([0.0])

def average_precision(preds: torch.Tensor, labels: torch.Tensor):
    """
    AP over a single group. See `get_mini_groups` for details about groups
    """
    labels = labels[torch.argsort(preds, dim=-1, descending=True)]
    positions = (torch.arange(len(labels), device=labels.device) + 1) * labels
    denominators = positions[torch.where(positions > 0)[0]]
    res = torch.true_divide((torch.arange(len(denominators), device=denominators.device) + 1), denominators).mean()
    return res

def precision(preds: torch.Tensor, labels: torch.Tensor, k: int = 1, empty_document=1.0):
    """
    Precision@k over a single group. See `get_mini_groups` for details about groups.
    Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    :param empty_document: what should be returned if document has no positive labels
    """
    assert preds.shape == labels.shape, (
        f"Predicions and labels must have the same shape, found {preds.shape} and {labels.shape} instead"
    )
    if labels.sum() == 0:
        return torch.tensor(empty_document).to(preds)
    return torch.true_divide(labels[torch.argsort(preds, dim=-1, descending=True)][:k].sum(), k).to(preds)

def recall(preds: torch.Tensor, labels: torch.Tensor, k: int = 1, empty_document=1.0):
    """
    Recall@k over a single group. See `get_mini_groups` for details about groups
    Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
    :param empty_document: what should be returned if document has no positive labels
    """
    assert preds.shape == labels.shape, (
        f"Predicions and labels must have the same shape, found {preds.shape} and {labels.shape} instead"
    )
    if labels.sum() == 0:
        return torch.tensor(empty_document).to(preds)
    return torch.true_divide(labels[torch.argsort(preds, dim=-1, descending=True)][:k].sum(), labels.sum()).to(preds)

def hit_rate(preds: torch.Tensor, labels: torch.Tensor, k: int = 1, empty_document=1.0):
    """
    HitRate@k over a single group. See `get_mini_groups` for details about groups
    HitRate@k = (# of recommended items @k that are relevant) > 0 ? 1 else 0
    :param empty_document: what should be returned if document has no positive labels
    """
    assert preds.shape == labels.shape, (
        f"Predicions and labels must have the same shape, found {preds.shape} and {labels.shape} instead"
    )
    if labels.sum() == 0:
        return torch.tensor(empty_document).to(preds)
    return (labels[torch.argsort(preds, dim=-1, descending=True)][:k].sum() > 0).to(preds)
