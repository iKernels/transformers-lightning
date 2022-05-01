import math
from argparse import ArgumentParser, Namespace

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import DataParallelStrategy, DDP2Strategy
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from transformers_lightning import optimizers, schedulers
from transformers_lightning.optimizers.super_optimizer import SuperOptimizer
from transformers_lightning.schedulers.super_scheduler import SuperScheduler
from transformers_lightning.utils.inspectors import get_classes_from_module

all_optimizers = get_classes_from_module(optimizers)
all_schedulers = get_classes_from_module(schedulers)


class TransformersModel(LightningModule):
    r"""
    `TransformersModel` add a ready-to-be-used optimizer and scheduler functions.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    config: PretrainedConfig
    hyperparameters: Namespace

    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        r"""
        Simply call the `model` attribute with the given args and kwargs
        """
        return self.model(*args, **kwargs)

    def get_optimizer(self) -> SuperOptimizer:
        r""" Get optimizer as defined by hyperparameters. """
        optim_class = all_optimizers[self.hyperparameters.optimizer_class]
        return optim_class(self.hyperparameters, self.named_parameters())

    def get_scheduler(self, optimizer) -> SuperScheduler:
        r""" Get scheduler as defined by hyperparameters. """
        sched_class = all_schedulers[self.hyperparameters.scheduler_class]
        return sched_class(self.hyperparameters, optimizer)

    def num_training_steps(self) -> int:
        r""" Total training steps inferred from datasets length, nodes and devices. """
        if self.trainer.max_steps is not None and self.trainer.max_steps >= 0:
            return self.trainer.max_steps

        if not has_len(self.trainer.datamodule.train_dataset):
            rank_zero_warn("Using IterableDataset, cannot compute max_steps, returning None")
            return None

        # train samples
        train_samples = len(self.trainer.datamodule.train_dataset)

        # number of training devices
        is_dataparallel = isinstance(self.trainer.strategy, (DataParallelStrategy, DDP2Strategy))
        if is_dataparallel:
            total_devices = self.trainer.num_nodes
        else:
            total_devices = self.trainer.num_devices * self.trainer.num_nodes

        # the number of training samples may be modified in distributed training
        # to be divisible by the number of GPUs...
        train_samples_per_device = math.ceil(train_samples / total_devices)

        # train batches from the dataloader
        train_batches_per_device = math.ceil(train_samples_per_device / self.hyperparameters.batch_size)

        # eventually limit train batches
        limit_batches = self.trainer.limit_train_batches
        train_batches_per_device = (
            min(train_batches_per_device, limit_batches)
            if isinstance(limit_batches, int) else int(limit_batches * train_batches_per_device)
        )

        # train steps for each device
        train_steps_per_device = math.ceil(train_batches_per_device / self.trainer.accumulate_grad_batches)

        # total train steps across all epochs
        total_train_steps = train_steps_per_device * self.trainer.max_epochs
        rank_zero_warn(f"Automatically computed total steps equal to {total_train_steps}")

        return total_train_steps

    def configure_optimizers(self):
        r"""
        Instantiate an optimizer on the parameters of self.model.
        A linear scheduler is also instantiated to manage the learning rate.
        """

        # fix max number of steps
        self.hyperparameters.max_steps = self.num_training_steps()

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,    # The LR schduler
                    'interval': self.hyperparameters.scheduler_interval,    # The unit of the scheduler's step size
                    'frequency': self.hyperparameters.scheduler_frequency,    # The frequency of the scheduler
                }
        }

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        parser.add_argument('--optimizer_class', type=str, default='AdamWOptimizer', choices=all_optimizers.keys())
        parser.add_argument(
            '--scheduler_class', type=str, default='LinearSchedulerWithWarmup', choices=all_schedulers.keys()
        )
        parser.add_argument('--scheduler_interval', type=str, default='step', choices=['step', 'epoch'])
        parser.add_argument('--scheduler_frequency', type=int, default=1)

        # retrieving model with temporary parsered arguments
        tmp_params, _ = parser.parse_known_args()

        # get pl_model_class in advance to know which params it needs, same for the datamodule
        optim_class = all_optimizers[tmp_params.optimizer_class]
        sched_class = all_schedulers[tmp_params.scheduler_class]

        # add optimizer and scheduler specific args
        optim_class.add_optimizer_specific_args(parser)
        sched_class.add_scheduler_specific_args(parser)
