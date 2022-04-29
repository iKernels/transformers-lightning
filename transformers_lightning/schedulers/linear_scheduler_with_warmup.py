from argparse import ArgumentParser, Namespace

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from transformers_lightning.schedulers.super_scheduler import SuperScheduler


class LinearSchedulerWithWarmup(SuperScheduler):
    r"""
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    More informations about the default parameters can be found on the documentation of
    `_LRScheduler` in the `torch` project.

    Args:
        hyperparameters: (:class:`~argparse.Namespace`):
            Collection of training hyperparameters.
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.

    Args through CLI:
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = LinearSchedulerWithWarmup(hyperparameters, optimizer)
    """

    def __init__(self, hyperparameters: Namespace, optimizer: torch.optim.Optimizer):
        super().__init__(hyperparameters, optimizer)

        if not isinstance(hyperparameters.num_warmup_steps, int) or not hyperparameters.num_warmup_steps >= 0:
            raise ValueError("`num_warmup_steps` must be an integer greater than 0")

    def lr_lambda(self, current_step: int) -> int:
        r""" Compute lambda that is going to scale the learning rate. """
        assert current_step <= self.num_training_steps

        if current_step < self.hyperparameters.num_warmup_steps:
            return float(current_step) / float(max(1, self.hyperparameters.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step) / float(
                max(1, self.num_training_steps - self.hyperparameters.num_warmup_steps)
            )
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            rank_zero_warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]

    @staticmethod
    def add_scheduler_specific_args(parser: ArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        super(LinearSchedulerWithWarmup, LinearSchedulerWithWarmup).add_scheduler_specific_args(parser)
        parser.add_argument('--num_warmup_steps', type=int, default=0)
