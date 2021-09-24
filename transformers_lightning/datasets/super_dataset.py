from argparse import Namespace

from pytorch_lightning.trainer.trainer import Trainer

from transformers_lightning.adapters.super_adapter import SuperAdapter


class SuperDataset:
    r"""
    Superclass of all datasets. Only implements some instantiations.
    """

    hyperparameters: Namespace = None
    adapter: SuperAdapter = None
    trainer: Trainer = None

    def __init__(
        self,
        hyperparameters: Namespace,
        adapter: SuperAdapter = None,
        trainer: Trainer = None,
        do_preprocessing: bool = True
    ):
        self.hyperparameters = hyperparameters
        self.adapter = adapter
        self.trainer = trainer
        self.do_preprocessing = do_preprocessing
