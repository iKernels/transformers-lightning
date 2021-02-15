import pytest
import torch

from transformers_lightning.metrics.retrieval import (
    MeanReciprocalRank, MeanAveragePrecision, Precision, Recall, HitRateAtK
)


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["ids", "predictions", "labels", "empty_docs", "k", "mrr", "map", "precision", "recall", "hit_rate"], [
        [
            torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2]),
            torch.tensor([False, False, True, False, True, False, False]), "skip", 1, 0.75, 0.75, 0.5, 0.5, 0.5
        ],
        [
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
            torch.tensor([0.2, 0.1, 0.3, 0.1, 0.7, 0.4, 0.3, 0.2, 0.1]),
            torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 0]), "skip", 1, 0.75, 0.75, 0.5, 0.25, 0.5
        ],
        [
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
            torch.tensor([0.2, 0.1, -0.3, 0.1, 0.7, 0.4, 0.3, 0.2, 0.1]),
            torch.tensor([1, 0, 1, 0, 0, 0, 0, 0, 0]), "pos", 2, 1.0, 0.875, 0.75, 0.75, 1.0
        ],
        [
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1]),
            torch.tensor([0.2, 0.1, -0.3, 0.1, 0.7, 0.4, 0.3, 0.2, 0.1]),
            torch.tensor([1, 0, 1, 0, 0, 0, 0, 0, 0]), "neg", 1, 0.5, 0.375, 0.5, 0.25, 0.5
        ],
    ]
)
def test_metrics(ids, predictions, labels, empty_docs, k, mrr, map, precision, recall, hit_rate):

    _mrr = MeanReciprocalRank(query_without_relevant_docs=empty_docs)
    _map = MeanAveragePrecision(query_without_relevant_docs=empty_docs)
    _prec = Precision(k=k, query_without_relevant_docs=empty_docs)
    _rec = Recall(k=k, query_without_relevant_docs=empty_docs)
    _hr = HitRateAtK(k=k, query_without_relevant_docs=empty_docs)

    _mrr = _mrr(ids, predictions, labels)
    _map = _map(ids, predictions, labels)
    _prec = _prec(ids, predictions, labels)
    _rec = _rec(ids, predictions, labels)
    _hr = _hr(ids, predictions, labels)

    assert _mrr == mrr, f"Computed mrr: {_mrr}, expected: {mrr}"
    assert _map == map, f"Computed map: {_map}, expected: {map}"
    assert _prec == precision, f"Computed precision: {_prec}, expected: {precision}"
    assert _rec == recall, f"Computed recall: {_rec}, expected: {recall}"
    assert _hr == hit_rate, f"Computed hit_rate: {_hr}, expected: {hit_rate}"
