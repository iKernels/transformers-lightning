import math

import torch
from torch.optim.lr_scheduler import _LRScheduler, warnings


class CosineSchedulerWithWarmup(_LRScheduler):
    r"""
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Example:
        >>> scheduler = ConstantSchedulerWithWarmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int = 0,
        num_training_steps: int = 0,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        if not isinstance(num_warmup_steps, int) or not num_warmup_steps >= 0:
            raise ValueError("`num_warmup_steps` must be an integer greater than 0")

        if not isinstance(num_training_steps, int) or not num_training_steps >= 0:
            raise ValueError("`num_warmup_steps` must be an integer greater than 0")

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles

        super().__init__(optimizer, last_epoch, verbose)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step -
                         self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
