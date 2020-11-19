from argparse import ArgumentParser
import math

from pytorch_lightning import LightningModule, _logger as logger
from transformers import AdamW
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

        try:
            dataset_len = len(self.trainer.datamodule.train_dataset)
        except:
            try:
                dataset_len = self.trainer.datamodule.train_dataset.length
            except:
                return None

        total_devices = utils.get_total_devices(trainer=self.trainer)

        num_training_batches = math.ceil(dataset_len / self.hparams.batch_size)
        training_batches_per_epoch = num_training_batches // total_devices
        steps_per_epoch = math.ceil(training_batches_per_epoch / self.hparams.accumulate_grad_batches)
        steps = self.hparams.max_epochs * steps_per_epoch

        logger.warning(
            f"Automatically computed max_steps={steps}. If it appears to be OK, ignore this warning"
        )

        return steps

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        """
        Add here parameters that you would like to add to the training session
        and return the parser.
        """
        return parser