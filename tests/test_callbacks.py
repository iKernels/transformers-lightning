import os
import random
import shutil
import string
from argparse import Namespace

import pytest
import pytorch_lightning as pl
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModelWithOptim, standard_args
from transformers_lightning.callbacks.transformers_model_checkpoint import TransformersModelCheckpointCallback


def random_name():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))


@pytest.mark.parametrize(
    "epochs, accumulate_grad_batches, batch_size, callback_interval, no_val_callback, expected_results", (
        [
            2, 3, 4, 3, True,
            [
                "hyperparameters.json", "ckpt_epoch_0_step_3", "ckpt_epoch_0_step_6", "ckpt_epoch_0_step_8",
                "ckpt_epoch_1_step_9", "ckpt_epoch_1_step_12", "ckpt_epoch_1_step_15", "ckpt_epoch_1_step_16_final"
            ]
        ],
        [1, 2, 5, 6, True, ["hyperparameters.json", "ckpt_epoch_0_step_6", "ckpt_epoch_0_step_10_final"]],
        [
            1, 2, 5, 6, False,
            [
                "hyperparameters.json", "ckpt_epoch_0_step_2", "ckpt_epoch_0_step_5", "ckpt_epoch_0_step_6",
                "ckpt_epoch_0_step_7", "ckpt_epoch_0_step_10_final"
            ]
        ],
    )
)
def test_model_checkpointing_callback(
    epochs, accumulate_grad_batches, batch_size, callback_interval, no_val_callback, expected_results
):

    hyperparameters = Namespace(
        batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=4,
        max_epochs=epochs,
        max_steps=-1,
        accelerator='cpu',
        iterable=False,
        checkpoint_interval=callback_interval,
        no_val_checkpointing=no_val_callback,
        no_epoch_checkpointing=False,
        pre_trained_dir='pre_trained_name',
        name=random_name(),
        val_check_interval=0.25,
        **standard_args,
    )

    tokenizer = BertTokenizer('tests/data/vocab.txt')
    callback = TransformersModelCheckpointCallback(hyperparameters)

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hyperparameters,
        profiler='simple',
        logger=None,
        callbacks=[callback],
    )

    # instantiate PL model
    model = DummyTransformerModelWithOptim(hyperparameters)

    # Datasets
    datamodule = DummyDataModule(hyperparameters, length_train=96, length_valid=96, length_test=96, tokenizer=tokenizer)

    model.datamodule = datamodule
    trainer.fit(model, datamodule=datamodule)

    folder = os.path.join(hyperparameters.output_dir, hyperparameters.pre_trained_dir, hyperparameters.name)
    listing = os.listdir(folder)
    shutil.rmtree(folder)
    assert set(listing) == set(expected_results), f"{sorted(listing)} vs {sorted(expected_results)}"
