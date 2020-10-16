import multiprocessing
from argparse import Namespace

import pytest
import pytorch_lightning as pl
import torch
import transformers_lightning
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer
from transformers.modeling_bert import (BertConfig,
                                        BertForSequenceClassification)

n_cpus = multiprocessing.cpu_count()

class SimpleTransformerLikeModel(transformers_lightning.models.SuperModel):

    def __init__(self, hparams):
        super().__init__(hparams)

        # super light BERT model
        config = BertConfig(hidden_size=12, num_hidden_layers=1, num_attention_heads=1, intermediate_size=12)
        self.model = BertForSequenceClassification(config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", config=config, cache_dir=hparams.cache_dir)

    def configure_optimizers(self):
        self.computed_max_steps = self.max_steps_anyway()
        return super().configure_optimizers()

    def training_step(self, batch, batch_idx):
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return { 'loss': results.loss, 'ids': batch['ids'] }

    def training_step_end(self, batch_parts):
        batch_parts['loss'] = torch.sum(batch_parts['loss'])
        return batch_parts


class ExampleDataModule(transformers_lightning.datamodules.SuperDataModule):

    def __init__(self, *args, ds_type=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_config = "dataset.yaml"
 
    train_dataloader = transformers_lightning.datamodules.SuperDataModule.default_train_dataloader

    

# Test if max_steps fix works correctly
@pytest.mark.parametrize(
    ["max_epochs", "accumulate_grad_batches", "batch_size", "distributed_backend", "gpus", "expected_max_steps"], [

    [1,             1,                         4,            None,                  None,   10],
    [1,             3,                         8,            None,                  None,   2],
    [4,             2,                         12,           None,                  None,   8],
    [4,             4,                         16,           None,                  None,   4],

])
def test_fix_max_steps_cpu(max_epochs, accumulate_grad_batches, batch_size, distributed_backend, gpus, expected_max_steps):

    hparams = Namespace(
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=4,
        dataset_dir='tests/test_data',
        config_dir='tests/test_data',
        cache_dir='cache',
        output_dir='output',
        max_epochs=max_epochs,
        max_steps=None,
        max_sequence_length=10,
        gpus=gpus,
        dataset_style='map'
    )

    if distributed_backend is not None:
        hparams.distributed_backend = distributed_backend

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        profiler=True,
        logger=None,
        callbacks=[],
    )

    # instantiate PL model
    model = SimpleTransformerLikeModel(hparams)    

    # Datasets
    datamodule = ExampleDataModule(hparams, model, trainer)

    # Train!
    if datamodule.do_train():
        trainer.fit(model, datamodule=datamodule)

    assert model.computed_max_steps == expected_max_steps


# Test if max_steps fix works correctly
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize(
    ["max_epochs", "accumulate_grad_batches", "batch_size", "distributed_backend", "gpus", "expected_max_steps"], [

    [1,             1,                         4,            'dp',                  1,   10],
    [1,             3,                         8,            'dp',                  2,   2],
    [4,             2,                         12,           'dp',                  2,   8],
    [4,             4,                         16,           'dp',                  2,   4],

    [1,             1,                         4,            'ddp',                 1,   5],
    [1,             3,                         8,            'ddp',                 2,   1],
    [4,             2,                         12,           'ddp',                 2,   4],
    [4,             4,                         16,           'ddp',                 2,   4],
])
def test_fix_max_steps_gpu(max_epochs, accumulate_grad_batches, batch_size, distributed_backend, gpus, expected_max_steps):

    hparams = Namespace(
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=4,
        dataset_dir='tests/test_data',
        config_dir='tests/test_data',
        cache_dir='cache',
        output_dir='output',
        max_epochs=max_epochs,
        max_steps=None,
        max_sequence_length=10,
        gpus=gpus,
        dataset_style='map'
    )

    if distributed_backend is not None:
        hparams.distributed_backend = distributed_backend

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        profiler=True,
        logger=None,
        callbacks=[],
    )

    # instantiate PL model
    model = SimpleTransformerLikeModel(hparams)    

    # Datasets
    datamodule = ExampleDataModule(hparams, model, trainer)

    # Train!
    if datamodule.do_train():
        trainer.fit(model, datamodule=datamodule)

    assert model.computed_max_steps == expected_max_steps