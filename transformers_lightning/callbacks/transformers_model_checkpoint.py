import os
import shutil
from argparse import ArgumentParser

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from transformers_lightning.utils import dump_json, is_simple

PARAMS_FILENAME = "hyperparameters.json"


class TransformersModelCheckpointCallback(Callback):
    r"""
        This class allow transformer-based models (inherited from the huggingface lib)
        to be saved and re-used with `--pre_trained_name` argument.

        Command line args:
        `--checkpoint_interval`: Save pre_trained models every given steps.
            A None value means save only at the end of each epoch.
        `--no_val_checkpointing`: Disable transformers checkpointing at each validation epoch end.
    """

    def __init__(self, hyperparameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparameters = hyperparameters
        self.destination = os.path.join(
            hyperparameters.output_dir, hyperparameters.pre_trained_dir, hyperparameters.name
        )

    def save_params(self):
        r"""
        Save a checkpoint of the training parameters (hyperparameters)
        This function is very useful to remeber the type of experiment among all the checkpoints
        """
        filepath = os.path.join(self.destination, PARAMS_FILENAME)
        dictionary = {k: v for k, v in vars(self.hyperparameters).items() if is_simple(v)}
        dump_json(filepath, dictionary, complain=False)

    def save_model(self, pl_module, epoch=None, step=None, final=False):
        r"""
        Called when the a checkpoint should be saved. Here models trained in the
        LightningModule will be saved to disk to be re-used.
        """
        basename = "ckpt"
        if epoch is not None:
            basename += f"_epoch_{epoch}"
        if step is not None:
            basename += f"_step_{step}"

            temporary_path = os.path.join(self.destination, basename)
            # it may happen that both step and epoch end try to save the same ckpt
            if os.path.isdir(temporary_path):
                shutil.rmtree(temporary_path)
        if final:
            basename += "_final"

        filepath = os.path.join(self.destination, basename)

        # create dest folder if it does not exist
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        # save models parts (Config, Model, Tokenizer) only if they are present
        if hasattr(pl_module, "config"):
            pl_module.config.save_pretrained(filepath)
        if hasattr(pl_module, "model"):
            pl_module.model.save_pretrained(filepath)
        if hasattr(pl_module, "tokenizer"):
            pl_module.tokenizer.save_pretrained(filepath)

    def on_train_start(self, trainer, pl_module):
        r""" Check model can be saved and save hyperparameters to understand what kind of experiment it was. """
        if trainer.global_rank != 0:
            return

        if not os.path.isdir(self.destination):
            os.makedirs(self.destination)

        if not hasattr(pl_module, "config"):
            rank_zero_warn(
                f"LightningModule {pl_module.__class__.__name__} has no "
                f"`config` attribute, then it will not be checkpointed."
            )
        if not hasattr(pl_module, "model"):
            rank_zero_warn(
                f"LightningModule {pl_module.__class__.__name__} has no "
                f"`model` attribute, then it will not be checkpointed."
            )
        if not hasattr(pl_module, "tokenizer"):
            rank_zero_warn(
                f"LightningModule {pl_module.__class__.__name__} has no "
                f"`tokenizer` attribute, then it will not be checkpointed."
            )

        self.save_params()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        r"""Called when the training batch ends. """
        # only run on main process
        if trainer.global_rank != 0:
            return

        # save only on last accumulated batch
        if ((batch_idx + 1) % trainer.accumulate_grad_batches) != 0:
            return

        # save only when global step is multiple of checkpoint_interval
        if (
            (self.hyperparameters.checkpoint_interval is None)
            or (trainer.global_step % self.hyperparameters.checkpoint_interval) != 0
        ):
            return

        self.save_model(pl_module, epoch=trainer.current_epoch, step=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        r"""Called when the train epoch ends."""
        # only run on main process
        if trainer.global_rank != 0:
            return

        # not epoch checkpointing if it is disabled
        if self.hyperparameters.no_epoch_checkpointing:
            return

        self.save_model(pl_module, epoch=trainer.current_epoch, step=trainer.global_step)

    def on_train_end(self, trainer, pl_module):
        r"""
        Called when the train ends. Here models trained in the
        LightningModule will be saved to disk to be re-used.
        """
        # only run on main process
        if trainer.global_rank != 0:
            return

        self.save_model(pl_module, epoch=trainer.current_epoch - 1, step=trainer.global_step, final=True)

    def on_validation_end(self, trainer, pl_module):
        r"""
        Called when the validation ends. Here models trained in the
        LightningModule will be saved to disk to be re-used.
        """
        # only run on main process
        if trainer.global_rank != 0:
            return

        # this probably was val_check control loop
        if trainer.global_step == 0:
            return

        # not validation checkpointing if it is disabled
        if self.hyperparameters.no_val_checkpointing:
            return

        self.save_model(pl_module, epoch=trainer.current_epoch, step=trainer.global_step)

    @staticmethod
    def add_callback_specific_args(parser: ArgumentParser):
        r""" Add callback_specific arguments to parser. """
        parser.add_argument(
            '--checkpoint_interval',
            type=int,
            required=False,
            default=None,
            help="Save pre_trained models every steps. A None value means save only at the end of each epoch."
        )
        parser.add_argument(
            '--no_val_checkpointing',
            action="store_true",
            help="Disable transformers checkpointing at each validation end."
        )
        parser.add_argument(
            '--no_epoch_checkpointing',
            action="store_true",
            help="Disable transformers checkpointing at the end of each epoch."
        )
        parser.add_argument(
            '--pre_trained_dir',
            type=str,
            required=False,
            default='pre_trained_models',
            help="Default path to save transformer models to."
        )
