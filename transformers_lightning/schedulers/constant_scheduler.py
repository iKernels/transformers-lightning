from argparse import Namespace

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from transformers_lightning.schedulers.super_scheduler import SuperScheduler


class ConstantScheduler(SuperScheduler):
    r"""
    Create a schedule with a learning rate that keeps a contant value.
    More informations about the default parameters can be found on the documentation of
    `_LRScheduler` in the `torch` project.

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.

    Args through CLI:
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = ConstantScheduler(hyperparameters, optimizer)
    """

    def __init__(self, hyperparameters: Namespace, optimizer: torch.optim.Optimizer):
        super().__init__(hyperparameters, optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            rank_zero_warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr for base_lr in self.base_lrs]
