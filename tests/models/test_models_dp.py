import pytest
import torch

from tests.models.helpers import do_test_fix_max_steps


@pytest.mark.parametrize("max_epochs", (1, 2))
@pytest.mark.parametrize("accumulate_grad_batches", (1, 3))
@pytest.mark.parametrize("batch_size" , (1, 2, 3, 8, 11))
@pytest.mark.parametrize("devices", (1, 2, 3))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_fix_max_steps_dp(max_epochs, accumulate_grad_batches, batch_size, devices):

    if torch.cuda.device_count() < devices:
        pytest.skip()

    do_test_fix_max_steps(
        max_epochs,
        accumulate_grad_batches,
        batch_size,
        devices=devices,
        strategy='dp',
        accelerator='gpu',
    )
