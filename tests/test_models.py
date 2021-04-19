from argparse import Namespace
from transformers_lightning import utils

import pytest
import torch
from pytorch_lightning import Trainer

from transformers import BertTokenizer, AdamW
from tests.helpers import DummyDataModule, DummyTransformerModel, standard_args


class TestTransformersModel(DummyTransformerModel):

    def configure_optimizers(self):
        r"""
        Instantiate an optimizer on the parameters of self.model.
        A linear scheduler is also instantiated to manage the learning rate.
        """
        # fix max number of steps
        self.computed_max_steps = utils.compute_max_steps(self.hparams, self.trainer)
        return AdamW(self.parameters())


class TestMaxStepsFix:

    @pytest.mark.parametrize(
        "max_epochs, accumulate_grad_batches, batch_size, expected_max_steps", (
            [1, 1, 4, 10],
            [1, 3, 8, 2],
            [4, 2, 12, 8],
            [4, 4, 16, 4],
            [1, 1, 4, 10],
            [1, 3, 8, 2],
            [4, 4, 16, 4],
            [4, 2, 12, 8],
        )
    )
    def test_fix_max_steps_cpu(self, max_epochs, accumulate_grad_batches, batch_size, expected_max_steps):

        self._do_test_fix_max_steps(
            max_epochs=max_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            batch_size=batch_size,
            expected_max_steps=expected_max_steps,
            gpus=None
        )

    # Test if max_steps fix works correctly
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires multi-GPU machine")
    @pytest.mark.parametrize(
        "max_epochs, accumulate_grad_batches, batch_size, distributed_backend, gpus, expected_max_steps", (
            [1, 1, 4, 'dp', 1, 10],
            [1, 3, 8, 'dp', 1, 2],
            [4, 2, 12, 'dp', 1, 8],
            [4, 4, 16, 'dp', 1, 4],
            [1, 1, 4, 'ddp', 1, 5],
            [1, 3, 8, 'ddp', 1, 1],
            [4, 2, 12, 'ddp', 1, 4],
            [4, 4, 16, 'ddp', 1, 4],
            [1, 1, 4, 'dp', 2, 10],
            [1, 3, 8, 'dp', 2, 2],
            [4, 2, 12, 'dp', 2, 8],
            [4, 4, 16, 'dp', 2, 4],
            [1, 1, 4, 'ddp', 2, 3],
            [1, 3, 8, 'ddp', 2, 1],
            [4, 2, 12, 'ddp', 2, 2],
            [4, 4, 16, 'ddp', 2, 2],
        )
    )
    def test_fix_max_steps_gpu(self, max_epochs, accumulate_grad_batches, batch_size, distributed_backend, gpus, expected_max_steps):

        if torch.cuda.device_count() < gpus:
            pytest.skip()

        self._do_test_fix_max_steps(
            max_epochs=max_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            batch_size=batch_size,
            expected_max_steps=expected_max_steps,
            gpus=gpus,
            accelerator=distributed_backend,
        )

    def _do_test_fix_max_steps(self, max_epochs, accumulate_grad_batches, batch_size, expected_max_steps, **kwargs):
        hparams = Namespace(
            batch_size=batch_size,
            val_batch_size=batch_size,
            test_batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            num_workers=4,
            max_epochs=max_epochs,
            max_steps=None,
            **standard_args,
            **kwargs
        )

        # instantiate PL trainer
        trainer = Trainer.from_argparse_args(hparams)

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = TestTransformersModel(hparams)

        # Datasets
        datamodule = DummyDataModule(hparams, tokenizer=tokenizer)
        model.datamodule = datamodule
        trainer.fit(model, datamodule=datamodule)

        # Assert max steps computed correctly
        assert model.computed_max_steps == expected_max_steps
