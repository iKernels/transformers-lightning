import math
from argparse import Namespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import _logger as logger
from pytorch_lightning.utilities.distributed import rank_zero_warn

from transformers_lightning import utils


def get_total_devices(trainer):
    r"""
    Compute total number of devices on which training is being performed 
    """
    if trainer.use_dp:
        return 1
    if trainer.use_ddp or trainer.use_ddp2:
        return torch.distributed.get_world_size()
    if trainer.on_gpu:
        return 1
    if trainer.on_tpu:
        return len(trainer.tpu_cores) * trainer.num_nodes
    if trainer.distributed_backend == 'ddp_cpu':
        return trainer.num_processes * trainer.num_nodes
    return 1


def compute_max_steps(hparams: Namespace, trainer: Trainer) -> int:
    r"""
    Compute total number of steps if not specified by the user.
    They may be required for example by eventual schedulers or optimizers.
    """

    # if already defined, skip
    if hparams.max_steps is not None:
        return hparams.max_steps

    if not hasattr(trainer, 'datamodule'):
        rank_zero_warn(
            "You tried to fix `max_steps` but didn't provide a datamodule to "
            "the trainer.fit function. Returning `max_steps=None`"
        )
        return None

    dataset_len = len(trainer.datamodule.train_dataset)
    total_devices = utils.get_total_devices(trainer=trainer)

    num_training_batches = math.ceil(dataset_len / hparams.batch_size)
    training_batches_per_epoch = num_training_batches // total_devices
    steps_per_epoch = math.ceil(training_batches_per_epoch / hparams.accumulate_grad_batches)
    steps = hparams.max_epochs * steps_per_epoch

    rank_zero_warn(f"Automagically computed max_steps={steps}. If it appears to be OK, ignore this warning")

    return steps
