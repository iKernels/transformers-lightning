import torch


def recall(
    preds: torch.Tensor,
    target: torch.Tensor,
    k: int = 1
):
    r"""
    Computes the recall @ k metric for information retrieval,
    as explained here: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(information_retrieval_context)
    Recall at K is the fraction of relevant documents in top K among all the relevant documents.

    `preds` and `target` should be of the same shape and live on the same device. If not target is true, 0 is returned.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant.
        k: consider only the top k elements.

    Returns:
        a single-value tensor with the recall at k (R@K) of the predictions `preds` wrt the labels `target`.

    Example:

        >>> preds = torch.tensor([0.2, 0.3, 0.5])
        >>> target = torch.tensor([True, False, True])
        >>> recall(preds, target, k=2)
        ... 0.5
    """

    if preds.shape != target.shape or preds.device != target.device: 
        raise ValueError(
            f"`preds` and `target` must have the same shape and be on the same device"
        )

    if target.sum() == 0:
        return torch.tensor([0]).to(preds)

    relevant = target[torch.argsort(preds, dim=-1, descending=True)][:k].sum()
    return torch.true_divide(relevant, target.sum())
