import pytest
import torch

from transformers_lightning.metrics.retrieval import (
    MeanReciprocalRank, MeanAveragePrecision, PrecisionAtK, RecallAtK, HitRateAtK
)


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["ids", "predictions", "labels", "empty_docs", "k", "mrr", "map", "precision", "recall", "hit_rate"], [
        [
            torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2]),
            torch.tensor([False, False, True, False, True, False, False]),
            "skip",
            1,
            0.75, 0.75, 0.5, 0.5, 0.5
        ],
        [
            torch.tensor([  0,  0,  0,  0,  1,  1,  1,  1,  1]),
            torch.tensor([  0.2,0.1,0.3,0.1,0.7,0.4,0.3,0.2,0.1]),
            torch.tensor([  1,  0,  1,  0,  0,  1,  0,  0,  0]),
            "skip",
            1,
            0.75, 0.75, 0.5, 0.25, 0.5
        ],
        # TODO: add tests
    ]
)
def test_metrics(ids, predictions, labels, empty_docs, k, mrr, map, precision, recall, hit_rate):

    _mrr = MeanReciprocalRank(empty_documents=empty_docs)
    _map = MeanAveragePrecision(empty_documents=empty_docs)
    _prec = PrecisionAtK(k=k, empty_documents=empty_docs)
    _rec = RecallAtK(k=k, empty_documents=empty_docs)
    _hr = HitRateAtK(k=k, empty_documents=empty_docs)

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
