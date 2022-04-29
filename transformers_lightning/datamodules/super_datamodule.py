import multiprocessing
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Callable

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torch.utils.data import DataLoader, Dataset

from transformers_lightning.utils.functional import collate_single_fn


class SuperDataModule(pl.LightningDataModule, ABC):
    r"""
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

    def __init__(self, hyperparameters: Namespace, trainer: Trainer, collate_fn: Callable = collate_single_fn):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.trainer = trainer
        self.collate_fn = collate_fn

    @abstractmethod
    def do_train(self) -> bool:
        r""" Whether to do training. """

    @abstractmethod
    def do_validation(self) -> bool:
        r""" Whether to do validation. """

    @abstractmethod
    def do_test(self):
        r""" Whether to do testing. """

    @abstractmethod
    def do_predict(self):
        r""" Whether to do predictions. """

    def default_dataloader(self, dataset: Dataset, batch_size: int, **kwargs):
        r""" Return a dataloader with all usual default parameters. """

        if 'sampler' in kwargs and kwargs['sampler'] is not None:
            rank_zero_warn(
                "Using a custom sampler may change the total number of steps, check model.num_training_steps"
            )
            if self.hyperparameters.replace_sampler_ddp is True:
                rank_zero_warn(
                    "You provided a custom sampler but lightning will override it."
                    " You should set --replace_sampler_ddp=False"
                )

        if self.hyperparameters.iterable and 'shuffle' in kwargs and kwargs['shuffle'] is True:
            raise ValueError(
                "Found shuffle=True while using IterableDataset"
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.hyperparameters.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def train_dataloader(self):
        r""" Return the training dataloader. """
        if self.do_train():
            params = dict(shuffle=True) if not self.hyperparameters.iterable else dict()
            return self.default_dataloader(self.train_dataset, self.hyperparameters.batch_size, **params)
        return None

    def val_dataloader(self):
        r""" Return the validation dataloader. """
        if self.do_validation():
            params = dict(shuffle=False) if not self.hyperparameters.iterable else dict()
            return self.default_dataloader(self.valid_dataset, self.hyperparameters.val_batch_size, **params)
        return None

    def test_dataloader(self):
        r""" Return the test dataloader. """
        if self.do_test():
            params = dict(shuffle=False) if not self.hyperparameters.iterable else dict()
            return [
                self.default_dataloader(dataset, self.hyperparameters.test_batch_size, **params)
                for dataset in self.test_dataset
            ]
        return None

    def predict_dataloader(self):
        r""" Return the validation dataloader. """
        if self.do_predict():
            params = dict(shuffle=False) if not self.hyperparameters.iterable else dict()
            return self.default_dataloader(self.predict_dataset, self.hyperparameters.predict_batch_size, **params)
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
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--val_batch_size', type=int, default=256)
        parser.add_argument('--test_batch_size', type=int, default=256)
        parser.add_argument('--predict_batch_size', type=int, default=256)
        parser.add_argument('--iterable', action="store_true")
