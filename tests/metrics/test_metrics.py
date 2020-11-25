import pytest
import torch
from transformers_lightning import metrics


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["ids", "predictions", "labels", "k", "expected_hit_rate", "expected_precision", "expected_recall"], [
    [
        torch.tensor([  0,  0,  0,  0,  1,  1,  1,  1,  1]),
        torch.tensor([  0.2,0.15,0.12,0.1,0.7,0.4,0.3,0.1,0.05]),
        torch.tensor([  1,  0,  1,  0,  0,  1,  0,  0,  0]),
        2,
        1.0, 0.5, 0.75
    ]
])
def test_id_metrics(ids, predictions, labels, k, expected_hit_rate, expected_precision, expected_recall):

    groups = metrics.get_mini_groups(ids)

    hit_rate = torch.stack([metrics.hit_rate(predictions[group], labels[group], k=k) for group in groups]).mean()
    precision = torch.stack([metrics.precision(predictions[group], labels[group], k=k) for group in groups]).mean()
    recall = torch.stack([metrics.recall(predictions[group], labels[group], k=k) for group in groups]).mean()

    assert hit_rate == expected_hit_rate, f"Computed: {hit_rate}, expected: {expected_hit_rate}"
    assert precision == expected_precision, f"Computed: {precision}, expected: {expected_precision}"
    assert recall == expected_recall, f"Computed: {recall}, expected: {expected_recall}"
