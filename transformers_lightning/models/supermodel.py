from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning import LightningModule
from transformers import AdamW


class SuperModel(LightningModule):
    r"""
    `SuperModel` is a super class that only implements a few method to help
    subclasses being lighter and to define a more standardized interface.
    """

    model: torch.nn.Module
    hparams: Namespace

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def configure_optimizers(self):
        return AdamW(self.parameters())

    def forward(self, *args, **kwargs) -> dict:
        r"""
        Simply call the `model` attribute with the given args and kwargs
        """
        return self.model(*args, **kwargs)

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        r"""
        Add here parameters that you would like to add to the training session
        and remember return the parser ;)
        """
        return parser
