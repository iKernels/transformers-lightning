import os

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from transformers_lightning.utils import (dump_json, get_version, is_simple,
                                          is_version, set_version)

PARAMS_FILENAME = "hparams.js"


class TransformersModelCheckpointCallback(Callback):
    r"""
        This class allow transformer-based models (inherited from the huggingface lib)
        to be save and re-used with `--pre_trained_name` argument.
        every_epochs: how frequenly save models in epochs, float accepted
        if both every_epochs and every_steps are defined, thei effects will be summed.
    """

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.destination = os.path.join(hparams.output_dir, hparams.pre_trained_dir, hparams.name)

    def init_folders(self):
        if not os.path.isdir(self.destination):
            os.makedirs(self.destination)

    def checkup(self, pl_module):
        assert hasattr(pl_module, "model"), f"pl_module must have a `model` attribute in order to save it"
        assert hasattr(pl_module, "tokenizer"), f"pl_module must have a `tokenizer` attribute in order to save it"

    def save_params(self):
        filepath = os.path.join(self.destination, PARAMS_FILENAME)
        dictionary = {k: v for k, v in vars(self.hparams).items() if is_simple(v)}
        dump_json(filepath, dictionary)

    def save_model(self, pl_module, step, epoch, final=False):
        """
        Called when the train ends. Here models trained in the
        LightningModule will be saved to disk to be re-used.
        """
        basename = f"model_at_epoch_{epoch}_step_{step}"
        if final:
            basename += "_final"

        filepath = os.path.join(self.destination, basename)

        # create dest folder if it does not exist
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        # config is automatically saved when model is saved
        pl_module.model.save_pretrained(filepath)
        pl_module.tokenizer.save_pretrained(filepath)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """ Check model can be saved and save hparams to understand what kind of experiment it was. """
        if trainer.global_rank != 0:
            return

        self.init_folders()
        self.checkup(pl_module)
        self.save_params()

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        """Called when the training batch ends. """
        # only run on main process
        if trainer.global_rank != 0:
            return

        step = pl_module.global_step
        epoch = trainer.current_epoch

        if (self.hparams.transformers_checkpoint_interval is not None
            and (step > 0)
            and (
                step % self.hparams.transformers_checkpoint_interval
            ) == 0
        ):
            # first compute the steps in this epoch and then if it is checkpoint time
            self.save_model(pl_module, step, epoch)

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        # only run on main process
        if trainer.global_rank != 0:
            return

        step = pl_module.global_step
        epoch = trainer.current_epoch

        self.save_model(pl_module, step, epoch)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """
        Called when the train ends. Here models trained in the
        LightningModule will be saved to disk to be re-used.
        """
        # only run on main process
        if trainer.global_rank != 0:
            return

        step = pl_module.global_step
        epoch = trainer.current_epoch

        self.save_model(pl_module, step, epoch, final=True)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        """
        Called when the validation ends. Here models trained in the
        LightningModule will be saved to disk to be re-used.
        """
        # only run on main process
        if trainer.global_rank != 0:
            return

        # not validation checkpointing if it is disabled
        if self.hparams.disable_val_checkpointing:
            return 

        step = pl_module.global_step
        epoch = trainer.current_epoch

        self.save_model(pl_module, step, epoch, final=True)

    @staticmethod
    def add_callback_specific_args(parser):
        """ Add callback_specific arguments to parser. """
        parser.add_argument('--transformers_checkpoint_interval', type=int, required=False, default=None,
                            help="Save pre_trained models every steps. A None value means save only at the end of each epoch.")
        parser.add_argument('--disable_val_checkpointing', action="store_true",
                            help="Disable transformers checkpointing at each validation end.")
        return parser
