import multiprocessing
from argparse import Namespace
import os

import pytest
import pytorch_lightning as pl
from .test_utils import SimpleTransformerLikeModel, ExampleDataModule

n_cpus = multiprocessing.cpu_count()


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["ds_type", "num_workers", "distributed_backend", "gpus", "epochs"], [
    
    # ITER dataset
    # test different num_workers in single node on cpu
    ['iter',     0,             None,                   0,   1],
    ['iter',     1,             None,                   0,   1],
    ['iter',     2,             None,                   0,   1],
    ['iter',     n_cpus,        None,                   0,   1],
    
    # num_workers through epochs
    ['iter',     0,             None,                   0,   1],
    ['iter',     0,             None,                   0,   2],
    ['iter',     0,             None,                   0,   4],
    ['iter',     0,             None,                   0,   10],
    ['iter',     2,             None,                   0,   1],
    ['iter',     2,             None,                   0,   2],
    ['iter',     2,             None,                   0,   4],

    # MAP dataset
    # test different num_workers in single node on cpu
    ['map',     0,             None,                   0,   1],
    ['map',     1,             None,                   0,   1],
    ['map',     2,             None,                   0,   1],
    ['map',     n_cpus,        None,                   0,   1],
    
    # num_workers through epochs
    ['map',     0,             None,                   0,   1],
    ['map',     0,             None,                   0,   2],
    ['map',     0,             None,                   0,   4],
    ['map',     2,             None,                   0,   1],
    ['map',     2,             None,                   0,   2],
    ['map',     2,             None,                   0,   4],
])
def test_datamodule_cpu(ds_type, num_workers, distributed_backend, gpus, epochs):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        dataset_style=ds_type
    )

    if distributed_backend is not None:
        hparams.distributed_backend = distributed_backend

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
    datamodule = ExampleDataModule(hparams, model, trainer)

    model.datamodule = datamodule
    # Train!
    if datamodule.do_train():
        trainer.fit(model, datamodule=datamodule)

    # Test!
    if datamodule.do_test():
        trainer.test(model, datamodule=datamodule)
