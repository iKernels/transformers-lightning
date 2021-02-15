from argparse import ArgumentParser
import multiprocessing
from typing import Iterable

from torch.utils.data.dataset import IterableDataset
from transformers_lightning.adapters import SuperAdapter

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers_lightning import utils
from transformers_lightning.datasets import (TransformersIterableDataset, TransformersMapDataset)


class SuperDataModule(pl.LightningDataModule):
    """
    SuperDataModule should be the parent class of all `LightingDataModules` in your project.
    It implements some simple methods to check whether training, val or testing is required.
    Example:

    >>> if datamodule.do_train():
    >>>     trainer.fit(model, datamodule=datamodule)
    
    >>> if datamodule.do_test():
    >>>     trainer.test(model, datamodule=datamodule)
    """

    train_adapter: SuperAdapter = None
    valid_adapter: SuperAdapter = None
    test_adapter: SuperAdapter = None

    def __init__(
        self, hparams, train_adapter=None, valid_adapter=None, test_adapter=None, collate_fn=utils.collate_single_fn
    ):
        super().__init__()
        self.hparams = hparams
        self.collate_fn = collate_fn

        # instantiate eventual adapters passed from init method
        if train_adapter is not None:
            assert isinstance(train_adapter, SuperAdapter), f"Argument `train_adapter` must be of type `SuperAdapter`"
            self.train_adapter = train_adapter

        if valid_adapter is not None:
            assert isinstance(valid_adapter, SuperAdapter), f"Argument `valid_adapter` must be of type `SuperAdapter`"
            self.valid_adapter = valid_adapter

        if test_adapter is not None:
            assert (
                isinstance(test_adapter, SuperAdapter) or isinstance(test_adapter, list)
            ), f"Argument `test_adapter` must be of type `SuperAdapter` or List[SuperAdapter]"

            if isinstance(test_adapter, list):
                for adapter in test_adapter:
                    assert isinstance(
                        adapter, SuperAdapter
                    ), (f"Argument `test_adapter` must be of type `SuperAdapter` or List[SuperAdapter]")
            self.test_adapter = test_adapter
        """
        This space should be used to instantiate the Adapters it they were not passed through the kwargs

        >>> self.train_adapter = TSVAdapter(self.hparams, "pre-training/train.tsv", delimiter="\t")
        >>> self.valid_adapter = TSVAdapter(self.hparams, "pre-training/valid.tsv", delimiter="\t")
        >>> self.test_adapter = TSVAdapter(self.hparams, "pre-training/test.tsv", delimiter="\t")
        """

    # Optional, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        """
        Load datasets only if respective Adapter are defined.
        Finally check that is a 
        This implementation should be enough for most subclasses.
        """
        if self.hparams.iterable_datasets:
            dataset_class = TransformersIterableDataset
        else:
            dataset_class = TransformersMapDataset

        if stage == 'fit' or stage is None:
            if self.train_adapter is not None:
                kwargs = {'start_from_step': self.hparams.skip_in_training} if self.hparams.iterable_datasets else {}
                self.train_dataset = dataset_class(self.hparams, self.train_adapter, self.trainer, **kwargs)
            if self.valid_adapter is not None:
                self.valid_dataset = dataset_class(self.hparams, self.valid_adapter, self.trainer)

            assert self.train_adapter is None or self.train_dataset is not None, (
                f"Cannot specify `train_adapter` and then `train_dataset` is None: "
                f"{self.train_adapter} and {self.train_dataset}"
            )
            assert self.valid_adapter is None or self.valid_dataset is not None, (
                f"Cannot specify `valid_adapter` and then `valid_dataset` is None: "
                f"{self.valid_adapter} and {self.valid_dataset}"
            )

        elif stage == 'test' or stage is None:
            if self.test_adapter is not None:
                if isinstance(self.test_adapter, SuperAdapter):
                    self.test_dataset = dataset_class(self.hparams, self.test_adapter, self.trainer)
                else:
                    self.test_dataset = [
                        dataset_class(self.hparams, adapter, self.trainer) for adapter in self.test_adapter
                    ]

            assert self.test_adapter is None or self.test_dataset is not None, (
                f"Cannot specify `test_adapter` and then `test_dataset` is None: "
                f"{self.test_adapter} and {self.test_dataset}"
            )

    def do_train(self):
        return self.train_adapter is not None

    def do_validation(self):
        return self.valid_adapter is not None

    def do_test(self):
        return self.test_adapter is not None

    def default_dataloader(self, dataset, batch_size, shuffle=False):
        """ Return a dataloader with all usual default parameters. """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=shuffle
        )

    def train_dataloader(self):
        if self.train_dataset:
            shuffle = not isinstance(self.train_dataset, IterableDataset)
            return self.default_dataloader(self.train_dataset, self.hparams.batch_size, shuffle=shuffle)
        return None

    def val_dataloader(self):
        if self.valid_adapter:
            return self.default_dataloader(self.valid_dataset, self.hparams.val_batch_size, shuffle=False)
        return None

    def test_dataloader(self):
        if self.test_adapter:
            if isinstance(self.test_dataset, list):
                return [
                    self.default_dataloader(dataset, self.hparams.test_batch_size, shuffle=False)
                    for dataset in self.test_dataset
                ]
            return self.default_dataloader(self.test_dataset, self.hparams.test_batch_size, shuffle=False)
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
        parser.add_argument('--skip_in_training', type=int, default=None, required=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--val_batch_size', type=int, default=256)
        parser.add_argument('--test_batch_size', type=int, default=256)
        parser.add_argument('--iterable_datasets', action="store_true")

        tmp_args, _ = parser.parse_known_args()
        assert tmp_args.skip_in_training is None or tmp_args.iterable_datasets, (
            f"At the moment, `--skip_in_training <steps>` can be used only with `--iterable_datasets`"
        )

        return parser
