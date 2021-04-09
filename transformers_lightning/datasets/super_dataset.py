from argparse import Namespace

from transformers_lightning.adapters.super_adapter import SuperAdapter


class SuperDataset:
    r"""
    Superclass of all datasets. Only implements hparams storage. 
    """

    def __init__(self, hparams: Namespace):
        self.hparams = hparams
