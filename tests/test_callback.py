import os
import shutil
from argparse import Namespace

import pytest
import pytorch_lightning as pl
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModel, standard_args
from transformers_lightning.callbacks.transformers_model_checkpoint import TransformersModelCheckpointCallback


# Test iter dataset work correctly
@pytest.mark.parametrize(
    "epochs, accumulate_grad_batches, batch_size, callback_interval, val_callback, expected_results", (
        [
            2, 3, 4, 3, False,
            [
                "hparams.json", "ckpt_epoch_0_step_3", "ckpt_epoch_0_step_6", "ckpt_epoch_0_step_8",
                "ckpt_epoch_1_step_9", "ckpt_epoch_1_step_12", "ckpt_epoch_1_step_15", "ckpt_epoch_1_step_16_final"
            ]
        ],
        [1, 2, 5, 6, False, ["hparams.json", "ckpt_epoch_0_step_6", "ckpt_epoch_0_step_10_final"]],
        [
            1, 2, 5, 6, True,
            [
                "hparams.json", "ckpt_epoch_0_step_1", "ckpt_epoch_0_step_3", "ckpt_epoch_0_step_5",
                "ckpt_epoch_0_step_6", "ckpt_epoch_0_step_8", "ckpt_epoch_0_step_10_final"
            ]
        ],
    )
)
def test_model_checkpointing_callback(epochs, accumulate_grad_batches, batch_size, callback_interval, val_callback, expected_results):

    hparams = Namespace(
        batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=4,
        max_epochs=epochs,
        max_steps=None,
        gpus=0,
        checkpoint_interval=callback_interval,
        no_val_checkpointing=not val_callback,
        no_epoch_checkpointing=False,
        pre_trained_dir='pre_trained_name',
        name="test",
        val_check_interval=0.25,
        **standard_args,
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    callback = TransformersModelCheckpointCallback(hparams)

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        profiler='simple',
        logger=None,
        callbacks=[callback],
    )

    # instantiate PL model
    model = DummyTransformerModel(hparams)

    # Datasets
    datamodule = DummyDataModule(hparams, test_number=2, tokenizer=tokenizer)

    model.datamodule = datamodule
    trainer.fit(model, datamodule=datamodule)

    folder = os.path.join(hparams.output_dir, hparams.pre_trained_dir, hparams.name)
    listing = os.listdir(folder)
    shutil.rmtree(hparams.output_dir)
    assert set(listing) == set(expected_results), f"{listing} vs {set(expected_results)}"
