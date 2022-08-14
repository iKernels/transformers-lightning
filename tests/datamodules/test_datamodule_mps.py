import multiprocessing

import pytest
import torch

from tests.datamodules.helpers import do_test_datamodule


@pytest.mark.parametrize("num_workers", [0, 1, 2, multiprocessing.cpu_count()])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 11])
@pytest.mark.parametrize("iterable", [False, True])
@pytest.mark.skipif(not torch.has_mps, reason="Skipping MPS tests because this machine has no MPS device")
def test_datamodule_mps(num_workers, batch_size, accumulate_grad_batches, iterable):

    do_test_datamodule(
        num_workers,
        batch_size,
        accumulate_grad_batches,
        iterable,
        accelerator='mps',
        devices=1,
        num_sanity_val_steps=0,
    )
