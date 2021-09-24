import pytest
import torch
from transformers_lightning.metrics import HitRate


@pytest.mark.parametrize(
    "metric_class, metric_kwargs, preds, target, idx, expected_result", (
        [
            HitRate,
            {'k': 1},
            torch.tensor([0.3, 0.5, 0.4, 0.9, 0.2], dtype=torch.float),
            torch.tensor([False, False, True, True, False], dtype=torch.bool),
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            torch.tensor(0.5)
        ],
        [
            HitRate,
            {'k': 1},
            torch.tensor([0.5, 0.2, 0.1, -0.6, 0.8], dtype=torch.float),
            torch.tensor([False, False, True, True, False], dtype=torch.bool),
            torch.tensor([0, 0, 1, 1, 0], dtype=torch.long),
            torch.tensor(0.5)
        ],
        [
            HitRate,
            {'k': 2},
            torch.tensor([0.3, 0.7, 0.5, 0.4, 0.9, 0.2], dtype=torch.float),
            torch.tensor([False, True, False, False, True, False], dtype=torch.bool),
            torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long),
            torch.tensor(1.0)
        ],
        [
            HitRate,
            {'k': 1},
            torch.tensor([0.3, 0.7, 0.5, 0.4, 0.9, 0.2], dtype=torch.float),
            torch.tensor([False, False, True, True, False, False], dtype=torch.bool),
            torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            torch.tensor(0.0)
        ],
        [
            HitRate,
            {'k': 2},
            torch.tensor([0.3, 0.7, 0.5, 0.4, 0.9, 0.2, 0.4, 0.5, 0.0], dtype=torch.float),
            torch.tensor([True, False, True, False, False, False, False, True, False], dtype=torch.bool),
            torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long),
            torch.tensor(2/3)
        ],
    )
)
def test_metrics(metric_class, metric_kwargs, preds, target, idx, expected_result):

    metric = metric_class(**metric_kwargs)
    metric.update(preds, target, idx)

    value = metric.compute()

    assert torch.allclose(
        expected_result,
        value,
    ), f"{expected_result} vs {value}"
