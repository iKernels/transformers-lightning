import os
import random
import shutil
import string
from argparse import Namespace

import pytorch_lightning as pl
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModel, standard_args
from transformers_lightning.loggers.jsonboard_logger import JsonBoardLogger


def random_name():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))


class DummyTransformerModelWithLogging(DummyTransformerModel):

    def training_step(self, batch, batch_idx):
        res = super().training_step(batch, batch_idx)
        self.log('training/loss', res['loss'], on_step=True)
        return res

    def validation_step(self, batch, batch_idx):
        res = super().validation_step(batch, batch_idx)
        self.log('validation/loss', res['loss'])
        return res

    def test_step(self, batch, batch_idx):
        res = super().test_step(batch, batch_idx)
        self.log('test/loss', res['loss'])
        return res


def test_jsonboard_logger():

    hyperparameters = Namespace(
        batch_size=3,
        val_batch_size=3,
        test_batch_size=4,
        accumulate_grad_batches=2,
        num_workers=4,
        max_epochs=1,
        max_steps=-1,
        accelerator='cpu',
        iterable=False,
        jsonboard_dir='jsonboard',
        name=random_name(),
        val_check_interval=0.25,
        log_every_n_steps=1,
        **standard_args,
    )

    tokenizer = BertTokenizer('tests/data/vocab.txt')
    logger = JsonBoardLogger(hyperparameters)

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hyperparameters,
        logger=logger,
        enable_checkpointing=False
    )

    # instantiate PL model
    model = DummyTransformerModelWithLogging(hyperparameters)

    # Datasets
    datamodule = DummyDataModule(hyperparameters, length_train=96, length_valid=96, length_test=96, tokenizer=tokenizer)

    model.datamodule = datamodule
    trainer.fit(model, datamodule=datamodule)

    folder = os.path.join(hyperparameters.output_dir, hyperparameters.jsonboard_dir, hyperparameters.name)
    assert set(os.listdir(logger.log_dir)) == set(['data.jsonl', 'hparams.json'])
    shutil.rmtree(folder)
