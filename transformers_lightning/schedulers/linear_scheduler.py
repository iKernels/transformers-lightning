import torch
from torch.optim.lr_scheduler import _LRScheduler, warnings


class LinearScheduler(_LRScheduler):
    r"""
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0.
    More informations about the default parameters can be found on the documentation of `_LRScheduler` in the `torch` project.
    
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = LinearScheduler(optimizer, num_training_steps=100)
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int, last_epoch: int = -1, verbose: bool = False
    ):
        if not isinstance(num_training_steps, int) or not num_training_steps >= 0:
            raise ValueError("`num_training_steps` must be an integer greater than 0")

        self.num_training_steps = num_training_steps

        super().__init__(optimizer, last_epoch, verbose)

    def lr_lambda(self, current_step: int) -> int:
        return max(0.0, float(self.num_training_steps - current_step) / self.num_training_steps)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
