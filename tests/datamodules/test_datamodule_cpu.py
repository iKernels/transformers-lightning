import multiprocessing
from argparse import Namespace
import os
from transformers import BertTokenizer

import pytest
import pytorch_lightning as pl
from .test_utils import SimpleTransformerLikeModel, ExampleDataModule

n_cpus = multiprocessing.cpu_count()


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["ds_type", "num_workers", "gpus", "epochs"],
    [

    # ITER dataset
    # test different num_workers in single node on cpu
        ['iter', 0, 0, 1],
        ['iter', 1, 0, 1],
        ['iter', 2, 0, 1],
        ['iter', n_cpus, 0, 1],

    # num_workers through epochs
        ['iter', 0, 0, 1],
        ['iter', 0, 0, 4],
        ['iter', 0, 0, 2],
        ['iter', 0, 0, 10],
        ['iter', 2, 0, 1],
        ['iter', 2, 0, 2],
        ['iter', 2, 0, 4],

    # MAP dataset
    # test different num_workers in single node on cpu
        ['map', 0, 0, 1],
        ['map', 1, 0, 1],
        ['map', 2, 0, 1],
        ['map', n_cpus, 0, 1],

    # num_workers through epochs
        ['map', 0, 0, 1],
        ['map', 0, 0, 2],
        ['map', 0, 0, 4],
        ['map', 2, 0, 1],
        ['map', 2, 0, 2],
        ['map', 2, 0, 4],
    ]
)
def test_datamodule_cpu(ds_type, num_workers, gpus, epochs):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        iterable_datasets=ds_type == 'iter',
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
