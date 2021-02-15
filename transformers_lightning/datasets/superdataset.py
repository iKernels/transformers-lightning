from argparse import Namespace

from pytorch_lightning import Trainer
from transformers_lightning.adapters.super_adapter import SuperAdapter


class SuperTransformersDataset:
    r"""
    Superclass of all datasets. Only implements default fields filling. 
    """
    hparams: Namespace = None
    adapter: SuperAdapter = None
    trainer: Trainer = None

    def __init__(self, hparams: Namespace, adapter: SuperAdapter, trainer: Trainer):
        r"""
        Doing type check to avoid stupid errors related to arguments inversion
        """
        assert isinstance(hparams, Namespace), f"Argument `hparams` must be of type `Namespace`"
        assert isinstance(adapter, SuperAdapter), f"Argument `adapter` must be of type `SuperAdapter`"
        assert isinstance(trainer, Trainer), f"Argument `trainer` must be of type `Trainer`"

        self.hparams = hparams
        self.adapter = adapter
        self.trainer = trainer
