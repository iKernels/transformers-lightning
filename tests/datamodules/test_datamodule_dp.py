import multiprocessing
import os
import time
from argparse import Namespace

import pytest
import pytorch_lightning as pl
import torch
from transformers import BertTokenizer

from .test_utils import ExampleDataModule, SimpleTransformerLikeModel

n_cpus = multiprocessing.cpu_count()


# Test iter dataset work correctly with dp
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize(
    ["num_workers", "distributed_backend", "gpus", "epochs"],
    [
    # num_workers with dp
        [0, 'dp', 2, 2],
        [1, 'dp', 2, 2],
        [2, 'dp', 2, 2],
        [n_cpus, 'dp', 2, 2],
        [0, 'dp', 2, 1],
        [1, 'dp', 2, 2],
        [2, 'dp', 2, 4],
        [n_cpus, 'dp', 2, 5]
    ]
)
def test_datamodule_gpu_dp(num_workers, distributed_backend, gpus, epochs):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    time.sleep(2)    # sleep for 5 second to be sure area is clean

    hparams = Namespace(
        batch_size=4,
        val_batch_size=4,
        test_batch_size=4,
        accumulate_grad_batches=3,
        num_workers=num_workers,
        output_dir='output',
        max_epochs=epochs,
        max_steps=None,
        max_sequence_length=10,
        gpus=gpus,
        accelerator=distributed_backend,
        skip_in_training=None
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

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
    datamodule = ExampleDataModule(hparams, test_number=1, tokenizer=tokenizer)

    model.datamodule = datamodule
    trainer.fit(model, datamodule=datamodule)
