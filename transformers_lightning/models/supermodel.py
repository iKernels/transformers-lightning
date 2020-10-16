import math

from pytorch_lightning import LightningModule, _logger as logger
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers_lightning import utils


class SuperModel(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def configure_optimizers(self):
        return AdamW(self.parameters())

    def forward(self, *args, **kwargs) -> dict:
        """ Usual BERT call with mandatory input_ids, attention_mask, labels and optional token_type_ids. """
        return self.model(*args, **kwargs)

    def max_steps_anyway(self) -> int:
        """ Compute total number of steps if not specified. They are required by eventual schedulers. """
        # if already defined, skip
        if self.hparams.max_steps is not None:
            return self.hparams.max_steps

        if not hasattr(self.trainer, 'datamodule'):
            logger.warning(
                "You tried to fix max_steps but didn't provide a datamodule to "
                "the trainer.fit function, returning max_steps=None"
            )
            return None

        # if cannot retrieve len of the dataset, skip
        # this can happen with iterabledatasets
        if not (
            hasattr(self.trainer.datamodule.train_dataset, '__len__') or
            hasattr(self.trainer.datamodule.train_dataset, 'length')
        ):
            logger.warning(
                "Cannot infer dataset length from datamodule.train_dataset, returning max_steps=None"
            )
            return None

        if (
            hasattr(self.trainer.datamodule.train_dataset, 'length') and
            hasattr(self.hparams, "accumulate_samples") and
            self.hparams.accumulate_samples
        ):
            logger.warning(
                "Using --accumulate_samples could reduce real max_steps value wrt"
                " the one computed in this function"
            )

        dataset_len = len(self.trainer.datamodule.train_dataset)

        if self.trainer.on_gpu:
            total_devices = self.trainer.num_nodes * self.trainer.num_processes
        elif self.trainer.on_tpu:
            total_devices = len(self.trainer.tpu_cores) * self.trainer.num_nodes
        elif self.trainer.distributed_backend == 'ddp_cpu':
            total_devices = self.trainer.num_processes
        else:
            total_devices = 1

        num_training_batches = math.ceil(dataset_len / self.hparams.batch_size)
        training_batches_per_epoch = num_training_batches // total_devices
        steps_per_epoch = math.ceil(training_batches_per_epoch / self.hparams.accumulate_grad_batches)
        steps = self.hparams.max_epochs * steps_per_epoch

        logger.warning(
            f"Automatically computed max_steps={steps}. If it appears to be OK, ignore this warning"
        )

        return steps
