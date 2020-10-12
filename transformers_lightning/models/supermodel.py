from pytorch_lightning import LightningModule
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
