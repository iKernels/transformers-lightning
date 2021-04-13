import multiprocessing
import torch
from argparse import Namespace

import pytest
import pytorch_lightning as pl
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModel, standard_args


# Test iter dataset work correctly
@pytest.mark.parametrize("num_workers", [0, 1, 2, multiprocessing.cpu_count()])
@pytest.mark.parametrize("epochs", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 11])
class TestDataModule:

    def test_datamodule_cpu(self, num_workers, epochs, batch_size, accumulate_grad_batches):
        self._do_test_datamodule(
            num_workers=num_workers,
            epochs=epochs,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            accelerator=None,
        )

    @pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count()), reason="Skipping GPU tests because this machine has not GPUs")
    def test_datamodule_gpu(self, num_workers, epochs, batch_size, accumulate_grad_batches):
        self._do_test_datamodule(
            num_workers=num_workers,
            epochs=epochs,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            accelerator=None,
            gpus=1,
        )

    @pytest.mark.parametrize("accelerator", ["dp", "ddp"])
    @pytest.mark.parametrize("gpus", [1, 2, 8])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping GPU tests because this machine has not GPUs")
    def test_datamodule_gpu_ddp(self, num_workers, epochs, batch_size, accumulate_grad_batches, accelerator, gpus):

        # cannot do GPU training without enough devices
        if torch.cuda.device_count() < gpus:
            pytest.skip()

        self._do_test_datamodule(
            num_workers=num_workers,
            epochs=epochs,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            accelerator=accelerator,
            gpus=gpus,
        )

    def _do_test_datamodule(self, num_workers, epochs, batch_size, accumulate_grad_batches, accelerator, **kwargs):

        hparams = Namespace(
            batch_size=batch_size,
            val_batch_size=batch_size,
            test_batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            num_workers=num_workers,
            max_epochs=epochs,
            max_steps=None,
            accelerator=accelerator,
            **standard_args,
            **kwargs,
        )
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        # instantiate PL trainer and model
        trainer = pl.Trainer.from_argparse_args(hparams)
        model = DummyTransformerModel(hparams)

        # Datasets
        datamodule = DummyDataModule(hparams, test_number=1, tokenizer=tokenizer)
        model.datamodule = datamodule

        trainer.fit(model, datamodule=datamodule)
