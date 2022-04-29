import multiprocessing

import pytest

from tests.datamodules.helpers import do_test_datamodule


@pytest.mark.parametrize("num_workers", [0, 1, 2, multiprocessing.cpu_count()])
@pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 11])
@pytest.mark.parametrize("iterable", [False, True])
def test_datamodule_cpu(num_workers, batch_size, accumulate_grad_batches, iterable):

    do_test_datamodule(
        num_workers,
        batch_size,
        accumulate_grad_batches,
        iterable,
        accelerator="cpu",
        num_sanity_val_steps=0,
    )
