from argparse import ArgumentParser, Namespace

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class SuperScheduler(_LRScheduler):

    def __init__(self, hyperparameters: Namespace, optimizer: Optimizer):
        self.hyperparameters = hyperparameters
        super().__init__(
            optimizer, last_epoch=hyperparameters.scheduler_last_epoch, verbose=hyperparameters.scheduler_verbose
        )

    @property
    def num_training_steps(self):
        r"""Get the current voltage."""
        if self.hyperparameters.max_steps is not None and self.hyperparameters.max_steps > 0:
            return self.hyperparameters.max_steps
        else:
            raise ValueError(f'scheduler {self.__class__.__name__} needs `max_steps` to be defined')

    @staticmethod
    def add_scheduler_specific_args(parser: ArgumentParser):
        r""" Add here the hyperparameters specific of the scheduler like the number of warmup steps. """
        parser.add_argument('--scheduler_last_epoch', type=int, default=-1)
        parser.add_argument('--scheduler_verbose', action='store_true')
