import math
from argparse import Namespace

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.distributed import rank_zero_warn


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
    total_devices = len(trainer.accelerator_connector.parallel_devices)
    rank_zero_warn(
        f"`compute_max_steps` found a total of {total_devices} devices"
    )

    num_training_batches = math.ceil(dataset_len / hparams.batch_size)  # assume drop_last=True in dataloader
    training_batches_per_epoch = math.ceil(num_training_batches / total_devices)  # ddp
    steps_per_epoch = math.ceil(training_batches_per_epoch / hparams.accumulate_grad_batches)
    steps = hparams.max_epochs * steps_per_epoch

    rank_zero_warn(
        f"Automagically computed max_steps={steps}. If it appears to be OK, ignore this warning"
    )

    return steps
