import os
import re
from argparse import ArgumentParser
from typing import Optional

import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn


class AdvancedModelCheckpointCallback(ModelCheckpoint):
    r"""
    Save the model after every epoch if it improves.

    After training finishes, use :attr:`best_model_path` to retrieve the path to the
    best checkpoint file and :attr:`best_model_score` to retrieve its score.

    Args:
        filepath: path to save the model file.
            Can contain named formatting options to be auto-filled.

            Example::

                # custom path
                # saves a file like: my/path/epoch_0.ckpt
                >>> checkpoint_callback = AdvancedModelCheckpoint('my/path/')

                # save any arbitrary metrics like `val_loss`, etc. in name
                # saves a file like: my/path/epoch=2-val_loss=0.2_other_metric=0.3.ckpt
                >>> checkpoint_callback = AdvancedModelCheckpoint(
                ...     filepath='my/path/{epoch}-{val_loss:.2f}-{other_metric:.2f}'
                ... )

            Can also be set to `None`, then it will be set to default location
            during trainer construction.

        monitor: quantity to monitor.
        verbose: verbosity mode. Default: ``False``.
        save_last: always saves the model at the end of the epoch. Default: ``False``.
        checkpoint_save_interval: How often within one training epoch
            to check save a checkpoint. A float in `[0.0, 1.0]` is expected. Default: ``None``.
        save_top_k: if `save_top_k == k`,
            the best k models according to
            the quantity monitored will be saved.
            if ``save_top_k == 0``, no models are saved.
            if ``save_top_k == -1``, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if ``save_top_k >= 2`` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        mode: one of {auto, min, max}.
            If ``save_top_k != 0``, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if ``True``, then only the model's weights will be
            saved (``model.save_weights(filepath)``), else the full model
            is saved (``model.save(filepath)``).
        period: Interval (number of epochs) between checkpoints.
        every_steps: Interval (number of steps) between checkpoints,
            unconditionally performed regardless of validation results.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import AdvancedModelCheckpoint

        # saves checkpoints to 'my/path/' whenever 'val_loss' has a new min
        >>> checkpoint_callback = AdvancedModelCheckpoint(filepath='my/path/')
        >>> trainer = Trainer(checkpoint_callback=checkpoint_callback)

        # save epoch and val_loss in name
        # saves a file like: my/path/sample-mnist_epoch=02_val_loss=0.32.ckpt
        >>> checkpoint_callback = AdvancedModelCheckpoint(
        ...     filepath='my/path/sample-mnist_{epoch:02d}-{val_loss:.2f}'
        ... )

        # retrieve the best checkpoint after training
        checkpoint_callback = AdvancedModelCheckpoint(filepath='my/path/')
        trainer = Trainer(checkpoint_callback=checkpoint_callback)
        model = ...
        trainer.fit(model)
        checkpoint_callback.best_model_path

    """

    def __init__(self, hparams, *args, **kwargs):
        self.hparams = hparams
        self.destination = os.path.join(hparams.output_dir, hparams.checkpoints_dir, hparams.name)
        filepath = os.path.join(self.destination, '{epoch}-{step}')
        super().__init__(*args, filepath=filepath, **kwargs)

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        # only run on main process
        if trainer.global_rank != 0:
            return

        step = pl_module.global_step
        if self.hparams.every_steps_checkpoint is None or \
            (step == 0) or \
            (step % self.hparams.every_steps_checkpoint) != 0:
            # no models are saved in step 0 or
            # in a step that is not multiple of `self.hparams.every_steps_checkpoint`
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        metrics["step"] = step

        filepath = self.format_checkpoint_name(epoch, metrics)
        self._save_model(filepath, trainer, pl_module)

    @staticmethod
    def add_callback_specific_args(parent_parser):
        """ Add callback_specific arguments to parser. """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--save_top_k', required=False, default=None, type=int,
                            help="Save top K checkpoints with respect to val metric")
        parser.add_argument('--every_steps_checkpoint', required=False, default=None, type=int,
                            help="Inteval between which checkpoint should be saved independenly of epochs and validations")
        return parser
