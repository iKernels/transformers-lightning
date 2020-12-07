import torch


def reciprocal_rank(
    preds: torch.Tensor,
    target: torch.Tensor
):
    r"""
    Computes reciprocal rank metric for information retrieval,
    as explained here: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

    `preds` and `target` should be of the same shape and live on the same device. If not target is true, 0 is returned.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant.

    Returns:
        a single-value tensor with the reciprocal rank (RR) of the predictions `preds` wrt the labels `target`.

    Example:

        >>> preds = torch.tensor([0.2, 0.3, 0.5])
        >>> target = torch.tensor([False, True, False])
        >>> reciprocal_rank(preds, target)
        ... 0.5
    """

    if preds.shape != target.shape or preds.device != target.device: 
        raise ValueError(
            f"`preds` and `target` must have the same shape and be on the same device"
        )

    if target.sum() == 0:
        return torch.tensor(0).to(preds)

    target = target[torch.argsort(preds, dim=-1, descending=True)]
    position = torch.where(target == 1)[0]
    res = 1.0 / (position[0] + 1)
    return res
