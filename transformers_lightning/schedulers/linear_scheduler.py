from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from transformers_lightning.schedulers.super_scheduler import SuperScheduler


class LinearScheduler(SuperScheduler):
    r"""
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0.
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
        >>> scheduler = LinearScheduler(hyperparameters, optimizer)
    """

    def lr_lambda(self, current_step: int) -> int:
        return max(0.0, float(self.num_training_steps - current_step) / self.num_training_steps)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            rank_zero_warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.")

        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
