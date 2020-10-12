import math
import multiprocessing
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers_lightning import utils
from transformers_lightning.datasets import (TransformersIterableDataset,
                                             TransformersMapDataset)


class SuperDataModule(pl.LightningDataModule):
    """
    SuperDataModule should be the parent class of all `LightingDataModules` in your project.
    It implements some simple methods to check whether training, val or testing is required
    to do things like:

    >>> if datamodule.do_train():
    >>>     trainer.fit(model, datamodule=datamodule)
    
    >>> if datamodule.do_test():
    >>>     trainer.test(model, datamodule=datamodule)
    """

    def __init__(self, hparams, model, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hparams = hparams
        self.tokenizer = model.tokenizer
        self.trainer = trainer

        self.train_config = None
        self.val_config = None
        self.test_config = None

    def get_config(self, config_file):
        """ Load a config file from standard directory and check that file exists! """
        config_path = os.path.join(self.hparams.config_dir, "datasets", config_file)
        assert os.path.isfile(config_path), f"Specified config {config_path} does not exist!"
        return utils.load_yaml(config_path)

    @property
    def train_config(self):
        if not hasattr(self, '_train_config'):
            self._train_config = None
        return self._train_config

    @train_config.setter
    def train_config(self, config_file):
        """ Load config only is it is not None. """
        if config_file is not None:
            self._train_config = self.get_config(config_file)
        else:
            self._train_config = None

    @property
    def val_config(self):
        if not hasattr(self, '_val_config'):
            self._val_config = None
        return self._val_config

    @val_config.setter
    def val_config(self, config_file):
        """ Load config only is it is not None. """
        if config_file is not None:
            self._val_config = self.get_config(config_file)
        else:
            self._val_config = None

    @property
    def test_config(self):
        if not hasattr(self, '_test_config'):
            self._test_config = None
        return self._test_config

    @test_config.setter
    def test_config(self, config_file):
        """ Load config only is it is not None. """
        if config_file is not None:
            self._test_config = self.get_config(config_file)
        else:
            self._test_config = None

    # Optional, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # Optional, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        """
        Load dataloaders only when needed. 
        This implementation should be enough for all subclasses.
        """

        dataset_class = (
            TransformersMapDataset if self.hparams.dataset_style == 'map' else TransformersIterableDataset
        )

        if stage == 'fit' or stage is None:
            if self.train_config is not None:
                self.train_dataset = dataset_class(
                    self.hparams, self.tokenizer, self.train_config
                )
            if self.val_config is not None:
                self.val_dataset = dataset_class(
                    self.hparams, self.tokenizer, self.val_config
                )

        elif stage == 'test' or stage is None:
            if self.test_config is not None:
                self.test_dataset = dataset_class(
                    self.hparams, self.tokenizer, self.test_config
                )

    def do_train(self):
        return hasattr(self, 'train_dataloader')

    def do_validation(self):
        return hasattr(self, 'val_dataloader')

    def do_test(self):
        return hasattr(self, 'test_dataloader')

    def default_dataloader(self, dataset, batch_size):
        """ Return a dataloader with all usual default parameters. """
        return DataLoader(dataset,
                          batch_size=batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=utils.collate_single_fn)

    def default_train_dataloader(self):
        return self.default_dataloader(self.train_dataset, self.hparams.batch_size)

    def default_val_dataloader(self):
        return self.default_dataloader(self.val_dataset, self.hparams.val_batch_size)

    def default_test_dataloader(self):
        return self.default_dataloader(self.test_dataset, self.hparams.test_batch_size)

    @staticmethod
    def add_datamodule_specific_args(parser):

        parser.add_argument('--num_workers', required=False, default=multiprocessing.cpu_count(), type=int,
                            help='Number of workers to be used to load datasets')

        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--val_batch_size', type=int, default=256)
        parser.add_argument('--test_batch_size', type=int, default=256)
        parser.add_argument('--dataset_style', type=str, choices=['map', 'iter'], default='map')

        return parser
