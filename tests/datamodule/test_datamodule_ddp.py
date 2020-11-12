import multiprocessing
from argparse import Namespace

import pytest
import pytorch_lightning as pl
import torch
from .test_utils import SimpleTransformerLikeModel, ExampleDataModule

n_cpus = multiprocessing.cpu_count()


# Test iter dataset work correctly with dp
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize(
    ["ds_type", "num_workers", "distributed_backend", "gpus", "epochs", "dataset_idx"], [

    # ITER dataset
    # num_workers with ddp
    ['iter',     0,             'ddp',                  2,      1,      1],
    ['iter',     1,             'ddp',                  2,      2,      1],
    ['iter',     2,             'ddp',                  2,      2,      1],
    ['iter',     0,             'ddp',                  2,      1,      2],
    ['iter',     n_cpus,        'ddp',                  2,      10,     3],

    # MAP dataset
    # num_workers with ddp
    ['map',     0,             'ddp',                  2,      2,       1],
    ['map',     1,             'ddp',                  2,      2,       1],
    ['map',     2,             'ddp',                  2,      2,       1],
    ['map',     0,             'ddp',                  2,      1,       2],
    ['map',     n_cpus,        'ddp',                  2,      10,      3],
])
def test_datamodule_gpu_ddp_only(ds_type, num_workers, distributed_backend, gpus, epochs, dataset_idx):

    hparams = Namespace(
        batch_size=4,
        val_batch_size=4,
        test_batch_size=4,
        accumulate_grad_batches=3,
        num_workers=num_workers,
        dataset_dir='tests/test_data',
        config_dir='tests/test_data',
        cache_dir='cache',
        output_dir='output',
        max_epochs=epochs,
        max_steps=None,
        max_sequence_length=10,
        gpus=gpus,
        dataset_style=ds_type,
        distributed_backend=distributed_backend
    )

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        profiler='simple',
        logger=None,
        callbacks=[],
    )

    # instantiate PL model
    model = SimpleTransformerLikeModel(hparams)    

    # Datasets
    datamodule = ExampleDataModule(hparams, model, trainer, train_config=f"dataset{dataset_idx}.yaml")

    model.datamodule = datamodule
    trainer.fit(model, datamodule=datamodule)
