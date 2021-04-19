import multiprocessing
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers_lightning import utils
from transformers_lightning.adapters.super_adapter import SuperAdapter
from transformers_lightning.datamodules.super_datamodule import SuperDataModule
from transformers_lightning.datasets.map_dataset import MapDataset


class AdaptersDataModule(SuperDataModule):
    """
    AdaptersDataModule should be used when you want to read and tokenizer data on-the-fly.
    It implements some simple methods to check whether training, val or testing is required.
    It work with adapters: you could define them inside the `__init__()` method or pass them
    as arguments.
    """

    def __init__(
        self,
        hparams: Namespace,
        train_adapter: SuperAdapter = None,
        valid_adapter: SuperAdapter = None,
        test_adapter: SuperAdapter = None,
    ):
        super().__init__(hparams)
        self.hparams = hparams

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

        if stage == 'fit' or stage is None:
            if self.train_adapter is not None:
                self.train_dataset = MapDataset(self.hparams, self.train_adapter, self.trainer)
            if self.valid_adapter is not None:
                self.valid_dataset = MapDataset(self.hparams, self.valid_adapter, self.trainer)

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
                    self.test_adapter = [self.test_adapter]
                self.test_dataset = [MapDataset(self.hparams, adapter, self.trainer) for adapter in self.test_adapter]

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
