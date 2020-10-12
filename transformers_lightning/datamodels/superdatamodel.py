import pytorch_lightning as pl
import math
from torch.utils.data import DataLoader
from transformers_lightning import utils
import os
import multiprocessing
from transformers_lightning.datasets import SuperTransformersDataset


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
        self.validation_config = None
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
    def validation_config(self):
        if not hasattr(self, '_validation_config'):
            self._validation_config = None
        return self._validation_config

    @validation_config.setter
    def validation_config(self, config_file):
        """ Load config only is it is not None. """
        if config_file is not None:
            self._validation_config = self.get_config(config_file)
        else:
            self._validation_config = None

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
        pass

    def do_train(self):
        return hasattr(self, 'train_dataloader')

    def do_validation(self):
        return hasattr(self, 'val_dataloader')

    def do_test(self):
        return hasattr(self, 'test_dataloader')

    @staticmethod
    def add_datamodel_specific_arguments(parser):

        parser.add_argument('--num_workers', required=False, default=multiprocessing.cpu_count(), type=int,
                            help='Number of workers to be used to load datasets')

        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--val_batch_size', type=int, default=256)
        parser.add_argument('--test_batch_size', type=int, default=256)

        return parser