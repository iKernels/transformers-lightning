import pytest

from tests.models.helpers import do_test_fix_max_steps


@pytest.mark.parametrize(
    "max_epochs, accumulate_grad_batches, batch_size, expected_max_steps", (
        [1, 1, 4, 10],
        [1, 3, 8, 2],
        [4, 2, 12, 8],
        [4, 4, 16, 4],
        [1, 1, 4, 10],
        [1, 3, 8, 2],
        [4, 4, 16, 4],
        [4, 2, 12, 8],
    )
)
def test_fix_max_steps_cpu(max_epochs, accumulate_grad_batches, batch_size, expected_max_steps):

    do_test_fix_max_steps(
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        batch_size=batch_size,
        expected_max_steps=expected_max_steps,
        gpus=0,
        accelerator="cpu",
    )
