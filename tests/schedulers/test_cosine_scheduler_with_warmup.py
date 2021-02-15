from argparse import Namespace
import os
import torch

import pytest
import pytorch_lightning as pl
from transformers_lightning.models import TransformersModel
from transformers_lightning.adapters import SuperAdapter
from transformers_lightning.datamodules import SuperDataModule
from transformers_lightning.schedulers import CosineSchedulerWithWarmup
from transformers import AdamW


class FakeModel(TransformersModel):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = torch.nn.Linear(10, 10)
        self.lrs = []

    def training_step(self, batch, *args):
        return {'loss': self(batch['data']).sum(), 'lr': self.trainer.optimizers[0].__dict__['param_groups'][0]['lr']}

    def training_epoch_end(self, outputs, *args, **kwargs):
        self.lrs = [o['lr'] for o in outputs] + [self.trainer.optimizers[0].__dict__['param_groups'][0]['lr']]

    def configure_optimizers(self):
        # Define adam optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate)

        # init scheduler after optional fp16 to get rid of strange warning about optimizer and scheduler steps order
        scheduler = CosineSchedulerWithWarmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=self.hparams.max_steps,
            num_cycles=0.5,
            last_epoch=self.hparams.last_epoch
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


class FakeAdapter(SuperAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for x in range(self.hparams.batch_size * self.hparams.max_steps):
            yield x

    def preprocess_line(self, line: list) -> list:
        res = {'data': [float(line)] * 10}
        return res


class FakeDataModule(SuperDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_adapter = FakeAdapter(self.hparams)


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["num_training_steps", "last_epoch", "expected_lrs"], [
        [
            20, -1,
            [
                0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.989074, 0.956773, 0.904508, 0.834565, 0.75, 0.654508, 0.552264,
                0.447735, 0.345491, 0.25, 0.165435, 0.095491, 0.043227, 0.010926, 0.0
            ]
        ]
    ]
)
def test_datamodule_cpu(num_training_steps, last_epoch, expected_lrs):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    hparams = Namespace(
        batch_size=1,
        num_workers=0,
        output_dir='output',
        max_epochs=1,
        max_steps=num_training_steps,
        last_epoch=last_epoch,
        max_sequence_length=10,
        gpus=0,
        iterable_datasets=False,
        log_every_n_steps=1,
        learning_rate=1.0,
    )

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        profiler='simple',
        logger=None,
        callbacks=[],
    )

    # instantiate PL model
    model = FakeModel(hparams)

    # Datasets
    datamodule = FakeDataModule(hparams)

    # Fit
    trainer.fit(model, datamodule=datamodule)

    assert torch.tensor(expected_lrs,
                        dtype=torch.float32).allclose(torch.tensor(model.lrs, dtype=torch.float32),
                                                      rtol=1e-04), (f"{expected_lrs} vs {model.lrs}")
