import torch
from torch.optim.lr_scheduler import _LRScheduler, warnings


class ConstantSchedulerWithWarmup(_LRScheduler):
    r"""
    Create a schedule with a learning rate that keeps a contant value but increases linearly for the first `num_warmup_steps`.
    More informations about the default parameters can be found on the documentation of `_LRScheduler` in the `torch` project.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = ConstantSchedulerWithWarmup(optimizer, num_warmup_steps=100)
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, num_warmup_steps: int = 0, last_epoch: int = -1, verbose: bool = False
    ):
        if not isinstance(num_warmup_steps, int) or not num_warmup_steps >= 0:
            raise ValueError("`num_warmup_steps` must be an integer greater than 0")

        self.num_warmup_steps = num_warmup_steps

        super().__init__(optimizer, last_epoch, verbose)

    def lr_lambda(self, current_step: int) -> int:
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return 1.0

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
