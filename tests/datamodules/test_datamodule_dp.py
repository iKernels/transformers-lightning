import multiprocessing

import pytest
import torch

from tests.datamodules.helpers import do_test_datamodule


@pytest.mark.parametrize("num_workers", [0, 1, 2, multiprocessing.cpu_count()])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 11])
@pytest.mark.parametrize("iterable", [False, True])
@pytest.mark.parametrize("gpus", [1, 2, 4, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping GPU tests because this machine has not GPUs")
def test_datamodule_gpu_dp(num_workers, batch_size, accumulate_grad_batches, gpus, iterable):

    # cannot do GPU training without enough devices
    if torch.cuda.device_count() < gpus:
        pytest.skip()

    do_test_datamodule(
        num_workers=num_workers,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        accelerator='dp',
        gpus=gpus,
        iterable=iterable,
    )
