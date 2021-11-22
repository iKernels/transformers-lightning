import logging
import os
from argparse import ArgumentParser, Namespace
from typing import Callable

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.trainer.states import TrainerFn

from transformers_lightning.datamodules.super_datamodule import SuperDataModule
from transformers_lightning.datasets.compressed_dataset import CompressedDataset
from transformers_lightning.utils.functional import collate_single_fn

logger = logging.getLogger("pytorch_lightning")


class CompressedDataModule(SuperDataModule):
    r"""
    CompressedDataModule implements some simple methods to check whether training, val or testing is required.
    It uses fast `CompressedDictionary`s to store data and return them by decompressing on-the-fly.
    """

    train_filepath: str = None
    valid_filepath: str = None
    test_filepath: str = None
    predict_filepath: str = None

    def __init__(
        self,
        hyperparameters: Namespace,
        trainer: Trainer,
        collate_fn: Callable = collate_single_fn,
        **kwargs,
    ):
        super().__init__(hyperparameters, trainer, collate_fn)

        # instantiate eventual adapters passed from init method
        if hyperparameters.train_filepath is not None:
            if not os.path.isfile(hyperparameters.train_filepath):
                raise ValueError("Argument `train_filepath` is not a valid file")
            self.train_filepath = hyperparameters.train_filepath

        if hyperparameters.valid_filepath is not None:
            if not os.path.isfile(hyperparameters.valid_filepath):
                raise ValueError("Argument `valid_filepath` is not a valid file")
            self.valid_filepath = hyperparameters.valid_filepath

        if hyperparameters.test_filepath is not None:
            for test_file in hyperparameters.test_filepath:
                if not os.path.isfile(test_file):
                    raise ValueError("file `{test_file}` is not a valid test file")
            self.test_filepath = hyperparameters.test_filepath

        if hyperparameters.predict_filepath is not None:
            if not os.path.isfile(hyperparameters.predict_filepath):
                raise ValueError("Argument `predict_filepath` is not a valid file")
            self.predict_filepath = hyperparameters.predict_filepath

        for kwarg in kwargs:
            logger.warning(f'CompressedDataModule received unused parameter {kwarg}')

        assert self.hyperparameters.iterable is False, (
            f"Cannot use IterableDataset with CompressedDataModule"
        )

    # Optional, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        r"""
        Load datasets only if respective file is defined.
        This implementation should be enough for most subclasses.
        """
        if stage == TrainerFn.FITTING.value or TrainerFn.VALIDATING.value:
            if self.do_train():
                logger.info("Loading training dataset from CompressedDictionary...")
                self.train_dataset = CompressedDataset(self.hyperparameters, self.train_filepath)
            if self.do_validation():
                logger.info("Loading validation dataset from CompressedDictionary...")
                self.valid_dataset = CompressedDataset(self.hyperparameters, self.valid_filepath)

        elif stage == TrainerFn.TESTING.value:
            if self.do_test():
                logger.info("Loading test dataset from CompressedDictionary...")
                self.test_dataset = [
                    CompressedDataset(self.hyperparameters, filepath) for filepath in self.test_filepath
                ]

        elif stage == TrainerFn.PREDICTING.value:
            if self.do_predict():
                logger.info("Loading predict dataset from CompressedDictionary...")
                self.predict_dataset = CompressedDataset(self.hyperparameters, self.predict_filepath)

    def train_dataloader(self):
        r""" Return the training dataloader. """
        if self.do_train():
            return self.default_dataloader(self.train_dataset, self.hyperparameters.batch_size, shuffle=True)
        return None

    def val_dataloader(self):
        r""" Return the validation dataloader. """
        if self.do_validation():
            return self.default_dataloader(self.valid_dataset, self.hyperparameters.val_batch_size, shuffle=False)
        return None

    def test_dataloader(self):
        r""" Return the test dataloader. """
        if self.do_test():
            return [
                self.default_dataloader(dataset, self.hyperparameters.test_batch_size, shuffle=False)
                for dataset in self.test_dataset
            ]
        return None

    def predict_dataloader(self):
        r""" Return the validation dataloader. """
        if self.do_predict():
            return self.default_dataloader(self.predict_dataset, self.hyperparameters.predict_batch_size, shuffle=False)
        return None

    def do_train(self):
        return self.train_filepath is not None

    def do_validation(self):
        return self.valid_filepath is not None

    def do_test(self):
        return len(self.test_filepath) > 0

    def do_predict(self):
        return self.predict_filepath is not None

    @staticmethod
    def add_datamodule_specific_args(parser: ArgumentParser):
        super(CompressedDataModule, CompressedDataModule).add_datamodule_specific_args(parser)
        parser.add_argument('--train_filepath', type=str, required=False, default=None, help="Path to training file")
        parser.add_argument('--valid_filepath', type=str, required=False, default=None, help="Path to validation file")
        parser.add_argument(
            '--test_filepath', type=str, required=False, default=[], nargs='+', help="Path to test file(s)"
        )
        parser.add_argument('--predict_filepath', type=str, required=False, default=None, help="Path to predict file")
