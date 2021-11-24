from argparse import ArgumentParser, Namespace

import pytest
import pytorch_lightning as pl
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModel, standard_args
from transformers_lightning.optimizers import AdamWOptimizer, ElectraAdamWOptimizer


class OptimModel(DummyTransformerModel):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.lrs = []

    def training_step(self, batch, *args):
        return super().training_step(batch, *args)

    def configure_optimizers(self):
        # Define adam optimizer
        optimizer = self.hyperparameters.optimizer_class(self.hyperparameters, self.model.named_parameters())
        return optimizer


@pytest.mark.parametrize("optimizer_class", [AdamWOptimizer, ElectraAdamWOptimizer])
@pytest.mark.parametrize("batch_size", [1, 4, 11])
def test_optimizers(optimizer_class, batch_size):

    hyperparameters = Namespace(
        batch_size=batch_size,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=0,
        max_epochs=1,
        max_steps=20,
        accelerator='cpu',
        iterable=False,
        log_every_n_steps=1,
        accumulate_grad_batches=1,
        **standard_args,
    )

    del (
        hyperparameters.learning_rate, hyperparameters.weight_decay,
        hyperparameters.adam_epsilon, hyperparameters.adam_betas
    )

    parser = ArgumentParser()
    optimizer_class.add_optimizer_specific_args(parser)
    hyperparameters = Namespace(**vars(hyperparameters), **vars(parser.parse_args("")))

    hyperparameters.optimizer_class = optimizer_class

    tokenizer = BertTokenizer('tests/data/vocab.txt')

    # instantiate PL trainer and model
    trainer = pl.Trainer.from_argparse_args(hyperparameters)
    model = OptimModel(hyperparameters)

    # Datasets and Fit
    datamodule = DummyDataModule(hyperparameters, length_train=96, tokenizer=tokenizer)
    trainer.fit(model, datamodule=datamodule)
