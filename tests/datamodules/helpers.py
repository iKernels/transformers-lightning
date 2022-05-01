from argparse import Namespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.distributed import distributed_available
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModelWithOptim, standard_args


class DummyTransformersModelIDCheck(DummyTransformerModelWithOptim):

    def __init__(self, hyperparameters, train_len: int, valid_len: int, test_len: int):
        super().__init__(hyperparameters)
        self.train_len = train_len
        self.valid_len = valid_len
        self.test_len = test_len

    def general_epoch_end(self, outputs, length: int):
        ids = torch.cat([o['ids'] for o in outputs], dim=0)

        # in distributed mode collect ids from every process (gpu)
        if distributed_available():
            ids = self.all_gather(ids).view(-1)

        received = torch.zeros(length).to(dtype=bool)
        received[ids] = True

        # assert no duplicate element received
        assert len(set(ids.tolist())) == len(
            ids.tolist()
        ), (f"Received {len(ids.tolist())} ids but only"
            f" {len(set(ids.tolist()))} are unique: {ids}")
        # assert all elements received
        assert all(received), (f"({self.trainer.max_steps}) Received not all {len(received)} ids: {received}")

    def training_epoch_end(self, outputs):
        return self.general_epoch_end(outputs, self.train_len)

    def validation_epoch_end(self, outputs):
        return self.general_epoch_end(outputs, self.valid_len)

    def test_epoch_end(self, outputs):
        return self.general_epoch_end(outputs, self.test_len)


def do_test_datamodule(num_workers, batch_size, accumulate_grad_batches, iterable, **kwargs):

    hyperparameters = Namespace(
        batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=num_workers,
        max_epochs=1,
        max_steps=-1,
        iterable=iterable,
        **standard_args,
        **kwargs,
    )

    tokenizer = BertTokenizer('tests/data/vocab.txt')

    train_len = 96
    valid_len = 96
    test_len = 96

    # instantiate PL trainer and model
    trainer = pl.Trainer.from_argparse_args(hyperparameters)
    model = DummyTransformersModelIDCheck(hyperparameters, train_len, valid_len, test_len)

    # Datasets
    datamodule = DummyDataModule(
        hyperparameters,
        length_train=train_len,
        length_valid=valid_len,
        length_test=test_len,
        tokenizer=tokenizer
    )
    trainer.fit(model, datamodule=datamodule)
