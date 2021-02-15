import multiprocessing
from argparse import Namespace
import os
from transformers import BertTokenizer

import pytest
import pytorch_lightning as pl
from transformers_lightning import datamodules
from .test_utils import SimpleTransformerLikeModel, ExampleAdapter

n_cpus = multiprocessing.cpu_count()


class ExampleDataModule(datamodules.SuperDataModule):

    def __init__(self, *args, test_number=1, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_adapter = ExampleAdapter(
            self.hparams, f"tests/test_data/test{test_number}.tsv", delimiter="\t", tokenizer=tokenizer
        )
        self.valid_adapter = ExampleAdapter(
            self.hparams, f"tests/test_data/test{test_number}.tsv", delimiter="\t", tokenizer=tokenizer
        )
        self.test_adapter = [
            ExampleAdapter(self.hparams, f"tests/test_data/test{test_number}.tsv", delimiter="\t", tokenizer=tokenizer)
            for _ in range(2)
        ]


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["ds_type", "num_workers", "gpus", "epochs"],
    [

    # ITER dataset
    # test different num_workers in single node on cpu
        ['iter', 0, 0, 1],
        ['iter', n_cpus, 0, 1],

    # num_workers through epochs
        ['iter', 0, 0, 1],
        ['iter', 0, 0, 10],

    # MAP dataset
    # test different num_workers in single node on cpu
        ['map', 0, 0, 1],
        ['map', n_cpus, 0, 1],

    # num_workers through epochs
        ['map', 0, 0, 1],
        ['map', 2, 0, 1],
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
    trainer.test(model, datamodule=datamodule)
