from typing import Any, Callable, Optional

from torch import Tensor
from torchmetrics.retrieval.retrieval_metric import RetrievalMetric

from transformers_lightning.metrics.functional.hit_rate import retrieval_hit_rate


class HitRate(RetrievalMetric):
    """
    Computes Hit Rate @ k, where 1.0 is given to queries where a relevant document is in the top `k`.

    Works with binary target data. Accepts float predictions from a model output.

    Forward accepts:

    - ``preds`` (float tensor): ``(N, ...)``
    - ``target`` (long or bool tensor): ``(N, ...)``
    - ``indexes`` (long tensor): ``(N, ...)``

    ``indexes``, ``preds`` and ``target`` must have the same dimension.
    ``indexes`` indicate to which query a prediction belongs.
    Predictions will be first grouped by ``indexes`` and then `Precision` will be computed as the mean
    of the `Precision` over each query.

    Args:
        empty_target_action:
            Specify what to do with queries that do not have at least a positive ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects
            the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None
        k: consider only the top k elements for each query. default: None

    Example:
        >>> from torchmetrics import WBQARetrievalPrecision
        >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
        >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = tensor([False, False, True, False, True, False, True])
        >>> p2 = WBQARetrievalPrecision(k=2)
        >>> p2(preds, target, indexes=indexes)
        tensor(0.5000)
    """

    def __init__(
        self,
        empty_target_action: str = 'neg',
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        k: int = None
    ):
        super().__init__(
            empty_target_action=empty_target_action,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn
        )

        if k is None:
            raise ValueError("`k` has to be a positive integer")

        self.k = k

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return retrieval_hit_rate(preds=preds, target=target, k=self.k)
