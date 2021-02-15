import torch
from torch.optim.lr_scheduler import _LRScheduler, warnings


class ConstantScheduler(_LRScheduler):
    r"""
    Create a schedule with a learning rate that keeps a contant value.
    More informations about the default parameters can be found on the documentation of `_LRScheduler` in the `torch` project.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = ConstantScheduler(optimizer)
    """

    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1, verbose: bool = False):
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr for base_lr in self.base_lrs]
