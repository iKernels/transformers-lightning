import math
from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from transformers_lightning.schedulers.super_scheduler import SuperScheduler


class CosineSchedulerWithWarmupAndHardRestart(SuperScheduler):
    r"""
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.

    Args through CLI:
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = CosineSchedulerWithWarmupAndHardRestart(hyperparameters, optimizer)
    """

    def __init__(self, hyperparameters: Namespace, optimizer: torch.optim.Optimizer):
        super().__init__(hyperparameters, optimizer)

        if not isinstance(hyperparameters.num_warmup_steps, int) or not hyperparameters.num_warmup_steps >= 0:
            raise ValueError("`num_warmup_steps` must be an integer greater than 0")

    def lr_lambda(self, current_step):
        if current_step < self.hyperparameters.num_warmup_steps:
            return float(current_step) / float(max(1, self.hyperparameters.num_warmup_steps))
        progress = float(current_step - self.hyperparameters.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.hyperparameters.num_warmup_steps)
        )
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(self.hyperparameters.num_cycles) * progress) % 1.0))))

    def get_lr(self):
        if not self._get_lr_called_within_step:
            rank_zero_warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]

    @staticmethod
    def add_scheduler_specific_args(parser: ArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super(
            CosineSchedulerWithWarmupAndHardRestart, CosineSchedulerWithWarmupAndHardRestart
        ).add_scheduler_specific_args(parser)
        parser.add_argument('--num_warmup_steps', type=int, default=0)
        parser.add_argument('--num_cycles', type=float, default=1.0)
