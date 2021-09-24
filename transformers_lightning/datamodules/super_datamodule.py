import multiprocessing
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from transformers_lightning import utils


class SuperDataModule(pl.LightningDataModule):
    """
    SuperDataModule should be the superclass of all the DataModule in your project.
    It implements some simple methods to check whether training, val or testing is required.
    Moreover, it adds to the command line parameters the basic arguments used by Dataset,
    like `batch_size`, `val_batch_size`, `test_batch_size` and `num_workers`.

    Example:

    >>> if datamodule.do_train():
    >>>     trainer.fit(model, datamodule=datamodule)

    >>> if datamodule.do_test():
    >>>     trainer.test(model, datamodule=datamodule)
    """

    train_dataset: Dataset = None
    valid_dataset: Dataset = None
    test_dataset: Dataset = None
    predict_dataset: Dataset = None

    def __init__(self, hyperparameters: Namespace, collate_fn: Callable = utils.collate_single_fn):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.collate_fn = collate_fn

    @abstractmethod
    def do_train(self) -> bool:
        """ Whether to do training. """

    @abstractmethod
    def do_validation(self) -> bool:
        """ Whether to do validation. """

    @abstractmethod
    def do_test(self):
        """ Whether to do testing. """

    @abstractmethod
    def do_predict(self):
        """ Whether to do predictions. """

    def default_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        """ Return a dataloader with all usual default parameters. """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.hyperparameters.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=self.hyperparameters.drop_last,
        )

    def train_dataloader(self):
        """ Return the training dataloader. """
        if self.do_train():
            return self.default_dataloader(
                self.train_dataset, self.hyperparameters.batch_size, shuffle=not self.hyperparameters.iterable
            )
        return None

    def val_dataloader(self):
        """ Return the validation dataloader. """
        if self.do_validation():
            return self.default_dataloader(self.valid_dataset, self.hyperparameters.val_batch_size, shuffle=False)
        return None

    def test_dataloader(self):
        """ Return the test dataloader. """
        if self.do_test():
            return [
                self.default_dataloader(dataset, self.hyperparameters.test_batch_size, shuffle=False)
                for dataset in self.test_dataset
            ]
        return None

    def predict_dataloader(self):
        """ Return the validation dataloader. """
        if self.do_predict():
            return self.default_dataloader(self.predict_dataset, self.hyperparameters.predict_batch_size, shuffle=False)
        return None

    @staticmethod
    def add_datamodule_specific_args(parser: ArgumentParser):
        parser.add_argument(
            '--num_workers',
            required=False,
            default=multiprocessing.cpu_count(),
            type=int,
            help='Number of workers to be used to load datasets'
        )
        parser.add_argument('--drop_last', action="store_true")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--val_batch_size', type=int, default=256)
        parser.add_argument('--test_batch_size', type=int, default=256)
        parser.add_argument('--predict_batch_size', type=int, default=256)
        parser.add_argument('--iterable', action="store_true")
