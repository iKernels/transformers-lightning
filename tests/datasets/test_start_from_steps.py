import multiprocessing
from argparse import Namespace
from os import access
from tests import adapters
import time

import pytest
import pytorch_lightning as pl
import torch
from transformers_lightning import datamodules, models, adapters

n_cpus = multiprocessing.cpu_count()
N = 100

class SimpleTransformerLikeModel(models.SuperModel):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.lin = torch.nn.Linear(10, 10)

    def configure_optimizers(self):
        self.computed_max_steps = self.max_steps_anyway()
        return super().configure_optimizers()

    def training_step(self, batch, batch_idx):
        results = self.lin(batch['data']).mean()
        return { 'loss': results, 'ids': batch['ids'] }

    def training_epoch_end(self, outputs):
        self.all_received_ids = torch.cat([output['ids'] for output in outputs])

        if self.trainer.distributed_backend == "ddp":
            gather_ids = [torch.ones_like(self.all_received_ids) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gather_ids, self.all_received_ids)
            self.all_received_ids = torch.cat(gather_ids, dim=0)

        self.all_received_ids = self.all_received_ids.tolist()


class ExampleAdapter(adapters.SuperAdapter):

    def __init__(self, hparams):
        super().__init__(hparams)

    def __iter__(self):
        for i in range(N):
            yield (i, [1.0] * 10, torch.LongTensor(1))

    def preprocess_line(self, line: list) -> list:
        res = { 'ids': line[0], 'data': line[1], 'labels': line[2] }
        return res


class ExampleDataModule(datamodules.SuperDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_adapter = ExampleAdapter(self.hparams)        


# Test if skip_steps works correctly
@pytest.mark.parametrize(
    ["max_epochs", "accumulate_grad_batches", "batch_size", "skip", "expected_ids"], [
    [1,             1,                         4,            4,         list(range(16, 100))],
    [1,             3,                         8,            3,         list(range(72, 100))],
    [1,             2,                         5,           5,          list(range(50, 100))],
])
def test_skip_steps_cpu(max_epochs, accumulate_grad_batches, batch_size, skip, expected_ids):

    hparams = Namespace(
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=5,
        dataset_dir='tests/test_data',
        config_dir='tests/test_data',
        cache_dir='cache',
        output_dir='output',
        max_epochs=max_epochs,
        max_sequence_length=10,
        gpus=None,
        max_steps=None,
        iterable_datasets=True,
        skip_in_training=skip
    )

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        profiler='simple',
        logger=None,
        callbacks=[],
    )

    # instantiate PL model
    model = SimpleTransformerLikeModel(hparams)    

    # Datasets
    datamodule = ExampleDataModule(hparams)

    # Train!
    trainer.fit(model, datamodule=datamodule)
    assert sorted(model.all_received_ids) == sorted(expected_ids), f"{sorted(model.all_received_ids)} != {sorted(expected_ids)}"



# Test if skip_steps works correctly in ddp
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize(
    ["max_epochs", "accumulate_grad_batches", "batch_size", "skip", "expected_ids"], [
    [1,             1,                         4,            4,         list(range(32, 100))],
    [1,             3,                         8,            2,         list(range(96, 100))],
    [1,             2,                         7,            2,         list(range(28, 100))],
])
def test_skip_steps_gpu(max_epochs, accumulate_grad_batches, batch_size, skip, expected_ids):

    hparams = Namespace(
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=5,
        dataset_dir='tests/test_data',
        config_dir='tests/test_data',
        cache_dir='cache',
        output_dir='output',
        max_epochs=max_epochs,
        max_sequence_length=10,
        gpus=2,
        max_steps=None,
        iterable_datasets=True,
        skip_in_training=skip,
        accelerator='ddp'
    )

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        profiler='simple',
        logger=None,
        callbacks=[],
    )

    # instantiate PL model
    model = SimpleTransformerLikeModel(hparams)    

    # Datasets
    datamodule = ExampleDataModule(hparams)

    # Train!
    trainer.fit(model, datamodule=datamodule)
    assert sorted(model.all_received_ids) == sorted(expected_ids), f"{sorted(model.all_received_ids)} != {sorted(expected_ids)}"