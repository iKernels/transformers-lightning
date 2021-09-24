import pytest
import torch

from tests.models.helpers import do_test_fix_max_steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize(
    "max_epochs, accumulate_grad_batches, batch_size, gpus, expected_max_steps", (
        [1, 1, 4, 1, 10],
        [1, 3, 8, 1, 2],
        [4, 2, 12, 1, 8],
        [4, 4, 16, 1, 4],
        [1, 1, 4, 2, 10],
        [1, 3, 8, 2, 2],
        [4, 2, 12, 2, 8],
        [4, 4, 16, 2, 4],
        [4, 4, 4, 3, 12],
        [4, 3, 3, 3, 20],
        [4, 4, 4, 4, 12],
        [4, 3, 3, 4, 20],
    )
)
def test_fix_max_steps_dp(
    max_epochs, accumulate_grad_batches, batch_size, gpus, expected_max_steps
):

    if torch.cuda.device_count() < gpus:
        pytest.skip()

    do_test_fix_max_steps(
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        batch_size=batch_size,
        expected_max_steps=expected_max_steps,
        gpus=gpus,
        accelerator='dp',
    )
