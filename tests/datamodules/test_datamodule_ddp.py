import multiprocessing

import pytest
import torch

from tests.datamodules.helpers import do_test_datamodule


@pytest.mark.parametrize("num_workers", [0, 1, 2, multiprocessing.cpu_count()])
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("accumulate_grad_batches", [2, 11])
@pytest.mark.parametrize("iterable", [False, True])
@pytest.mark.parametrize("devices", [1, 2, 4, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping GPU tests because this machine has not GPUs")
def test_datamodule_gpu_ddp(num_workers, batch_size, accumulate_grad_batches, devices, iterable):

    # cannot do GPU training without enough devices
    if torch.cuda.device_count() < devices:
        pytest.skip()

    do_test_datamodule(
        num_workers,
        batch_size,
        accumulate_grad_batches,
        iterable,
        strategy='ddp',
        accelerator='gpu',
        devices=devices,
        num_sanity_val_steps=0,
    )

    torch.distributed.destroy_process_group()
