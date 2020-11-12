import multiprocessing
from argparse import Namespace

import pytest
import pytorch_lightning as pl
import torch
from transformers_lightning import models, datamodules
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer
from transformers.modeling_bert import (BertConfig,
                                        BertForSequenceClassification)
from .test_utils import SimpleTransformerLikeModel, ExampleDataModule

n_cpus = multiprocessing.cpu_count()


# Test iter dataset work correctly with dp
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize(
    ["ds_type", "num_workers", "distributed_backend", "gpus", "epochs"], [
    
    # ITER dataset
    # num_workers with dp
    ['iter',     0,             'dp',                   2,      2],
    ['iter',     1,             'dp',                   2,      2],
    ['iter',     2,             'dp',                   2,      2],
    ['iter',     n_cpus,        'dp',                   2,      2],

    ['iter',     0,             'dp',                   2,      1],
    ['iter',     1,             'dp',                   2,      2],
    ['iter',     2,             'dp',                   2,      4],
    ['iter',     n_cpus,        'dp',                   2,      10],

    # MAP dataset
    # num_workers with dp
    ['map',     0,             'dp',                   2,      2],
    ['map',     1,             'dp',                   2,      2],
    ['map',     2,             'dp',                   2,      2],
    ['map',     n_cpus,        'dp',                   2,      2],

    ['map',     0,             'dp',                   2,      1],
    ['map',     1,             'dp',                   2,      2],
    ['map',     2,             'dp',                   2,      4],
    ['map',     n_cpus,        'dp',                   2,      10]
])
def test_datamodule_gpu_dp(ds_type, num_workers, distributed_backend, gpus, epochs):
    
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
