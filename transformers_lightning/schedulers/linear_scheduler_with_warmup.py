import torch
from torch.optim.lr_scheduler import _LRScheduler, warnings


class LinearSchedulerWithWarmup(_LRScheduler):
    r"""
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    More informations about the default parameters can be found on the documentation of `_LRScheduler` in the `torch` project.
    
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = LinearSchedulerWithWarmup(optimizer, num_warmup_steps=10, num_training_steps=100)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        if not isinstance(num_training_steps, int) or not num_training_steps >= 0:
            raise ValueError("`num_training_steps` must be an integer greater than 0")

        if not isinstance(num_warmup_steps, int) or not num_warmup_steps >= 0:
            raise ValueError("`num_warmup_steps` must be an integer greater than 0")

        self._num_warmup_steps = num_warmup_steps
        self._num_training_steps = num_training_steps

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def lr_lambda(self, current_step: int) -> int:
        """ Compute lambda that is going to scale the learning rate. """

        assert current_step <= self._num_training_steps

        if current_step < self._num_warmup_steps:
            return float(current_step) / float(max(1, self._num_warmup_steps))
        return max(
            0.0,
            float(self._num_training_steps - current_step) /
            float(max(1, self._num_training_steps - self._num_warmup_steps))
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
