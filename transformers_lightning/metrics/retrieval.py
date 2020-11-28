import torch
from pytorch_lightning.metrics import Metric

from transformers_lightning.language_modeling import IGNORE_IDX
from transformers_lightning.metrics.utils import get_mini_groups, masked_metric
from transformers_lightning.metrics.functional import (
    reciprocal_rank, average_precision, precision, recall, hit_rate
)


class RetrievalMetric(Metric):
    r"""
    Compute a metric for Information Retrieval by grouping predictions on same
    document using indexes. Detailed information about metrics are contained
    in sub-classes.

    It may be possible that a document has no positive label: this case can
    be managed in different ways using the `empty_documents` parameter.
    - `error`: an error is raised
    - `skip`: those documents are skipped (default)
    - `positive`: those documents are counted as positive preds
    - `negative`: those documents are counted as negative preds

    Entries with labels equal to IGNORE_IDX will be ignored.
    Subclasses must override at least the `metric` method.
    """

    options = ['error', 'skip', 'positive', 'negative']

    def __init__(self, dist_sync_on_step=False, empty_documents='skip', exclude=IGNORE_IDX):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        assert empty_documents in self.options, (
            f"`empty_documents` received a wrong value {empty_documents}."
            f"Allowed values are {self.options}"
        )

        self.empty_documents = empty_documents
        self.exclude = exclude

        self.add_state("idx", default=torch.tensor([], dtype=torch.int64), dist_reduce_fx="cat")
        self.add_state("preds", default=torch.tensor([], dtype=torch.int64), dist_reduce_fx="cat")
        self.add_state("target", default=torch.tensor([], dtype=torch.int64), dist_reduce_fx="cat")

    def update(self, idx: torch.Tensor, preds: torch.Tensor, target: torch.Tensor):
        assert idx.shape == preds.shape == target.shape

        self.idx = torch.cat([self.idx, idx])
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])
    
    def compute(self):
        res = []
        for group in get_mini_groups(self.idx):
            if self.target[group].sum() == 0:
                if self.empty_documents == 'error':
                    raise ValueError(
                        f"MeanReciprocalRank was provided of a prediction with no positive values, idx: {group}"
                    )
                elif self.empty_documents == 'positive': res.append(torch.tensor([1.0]))
                elif self.empty_documents == 'negative': res.append(torch.tensor([0.0]))
            else:
                res.append(
                    self.metric(group)
                )
        return torch.stack(res).mean()

    def metric(self, group):
        r""" Compute a metric over a single group. """
        raise NotImplementedError("This method must be overridden by subclasses")


class MeanReciprocalRank(RetrievalMetric):
    r"""
    Mean Reciprocal Rank computes the MRR over multiple predictions.
    Each reciprocal rank computation can be done on a different number of predictions
    thanks to the usage of a tensor dedicated to indexes.

    Example:
    >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = torch.tensor([False, False, True, False, True, False, False])

    >>> mrr = MeanReciprocalRank()
    >>> mrr(indexes, preds, target)
    >>> mrr.compute()
    ... 0.75
    """

    def metric(self, group):
        return masked_metric(predictions=self.preds[group],
                             labels=self.target[group],
                             exclude=self.exclude,
                             metric=reciprocal_rank,
                             args=[],
                             kwargs={})

class MeanAveragePrecision(RetrievalMetric):
    r"""
    Mean Average Precision computes the MAP over multiple predictions.
    Each average precision computation can be done on a different number of predictions
    thanks to the usage of a tensor dedicated to indexes.

    Example:
    >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = torch.tensor([False, False, True, False, True, False, False])

    >>> map = MeanAveragePrecision()
    >>> map(indexes, preds, target)
    >>> map.compute()
    ... 0.75
    """

    def metric(self, group):
        return masked_metric(predictions=self.preds[group],
                             labels=self.target[group],
                             exclude=self.exclude,
                             metric=average_precision,
                             args=[],
                             kwargs={})


class PrecisionAtK(RetrievalMetric):
    r"""
    Precision at K computes the P@K over multiple predictions.
    Each precision at k computation can be done on a different number of predictions
    thanks to the usage of a tensor dedicated to indexes.

    Example:
    >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = torch.tensor([False, False, True, False, True, False, False])

    >>> p_k = PrecitionAtK(k=1)
    >>> p_k(indexes, preds, target)
    >>> p_k.compute()
    ... 0.5
    """

    def __init__(self, *args, k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def metric(self, group):
        return masked_metric(predictions=self.preds[group],
                             labels=self.target[group],
                             exclude=self.exclude,
                             metric=precision,
                             args=[],
                             kwargs={'k': self.k})


class RecallAtK(RetrievalMetric):
    r"""
    Recall at K computes the R@K over multiple predictions.
    Each recall at k computation can be done on a different number of predictions
    thanks to the usage of a tensor dedicated to indexes.

    Example:
    >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = torch.tensor([False, False, True, False, True, False, False])

    >>> r_k = RecallAtK(k=1)
    >>> r_k(indexes, preds, target)
    >>> r_k.compute()
    ... 0.5
    """

    def __init__(self, *args, k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def metric(self, group):
        return masked_metric(predictions=self.preds[group],
                             labels=self.target[group],
                             exclude=self.exclude,
                             metric=recall,
                             args=[],
                             kwargs={'k': self.k})


class HitRateAtK(RetrievalMetric):
    r"""
    Hit Rate at K computes the HR@K over multiple predictions.
    Each hit rate at k computation can be done on a different number of predictions
    thanks to the usage of a tensor dedicated to indexes.
    Notice that HR@1 == P@1

    Example:
    >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = torch.tensor([False, False, True, False, True, False, False])

    >>> hr_k = HitRateAtK(k=1)
    >>> hr_k(indexes, preds, target)
    >>> hr_k.compute()
    ... 0.5
    """

    def __init__(self, *args, k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def metric(self, group):
        return masked_metric(predictions=self.preds[group],
                             labels=self.target[group],
                             exclude=self.exclude,
                             metric=hit_rate,
                             args=[],
                             kwargs={'k': self.k})
