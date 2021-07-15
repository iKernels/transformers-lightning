from argparse import Namespace

from pytorch_lightning.trainer.trainer import Trainer

from transformers_lightning.adapters.super_adapter import SuperAdapter


class SuperDataset:
    r"""
    Superclass of all datasets. Only implements some instantiations. 
    """

    hparams: Namespace = None
    adapter: SuperAdapter = None
    trainer: Trainer = None

    def __init__(self, hparams: Namespace, adapter: SuperAdapter = None, trainer: Trainer = None):
        self.hparams = hparams
        self.adapter = adapter
        self.trainer = trainer
