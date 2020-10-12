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

    def training_step(self, batch, batch_idx):
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return { 'loss': results.loss, 'ids': batch['ids'] }

    def training_step_end(self, batch_parts):
        batch_parts['loss'] = torch.sum(batch_parts['loss'])
        return batch_parts

    def training_epoch_end(self, outputs):
        ids = torch.cat([o['ids'] for o in outputs], dim=0)
        try:
            received = torch.zeros((len(self.datamodule.train_dataset),))
        except TypeError:
            received = torch.zeros((self.datamodule.train_dataset.length,))
        received[ids] = True
        assert all(received), f"({self.trainer.max_steps}) Received not all {len(received)} ids: {received}"

    def validation_step(self, batch, batch_idx):
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return results.loss

    def test_step(self, batch, batch_idx):
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return results.loss


class ExampleDataModule(transformers_lightning.datamodels.SuperDataModule):

    def __init__(self, *args, ds_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds_type = ds_type

        self.train_config = "dataset.yaml"
 
    # Optional, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        if self.ds_type == 'iter':
            dataset_class = transformers_lightning.datasets.TransformersIterableDataset
        elif self.ds_type == 'map':
            dataset_class = transformers_lightning.datasets.TransformersMapDataset
        else:
            raise ValueError(f"ds_type {self.ds_type} not recognized")

        self.train_dataset = dataset_class(
            self.hparams, self.tokenizer, self.train_config
        )

    def train_dataloader(self):
        dataset = DataLoader(self.train_dataset,
                             batch_size=self.hparams.batch_size,
                             num_workers=self.hparams.num_workers,
                             pin_memory=True,
                             collate_fn=transformers_lightning.utils.collate_single_fn)
        return dataset


# Test iter dataset work correctly
@pytest.mark.parametrize(
    ["ds_type", "num_workers", "distributed_backend", "gpus", "epochs"], [
    
    # ITER dataset
    # test different num_workers in single node on cpu
    ['iter',     0,             None,                   0,   1],
    ['iter',     1,             None,                   0,   1],
    ['iter',     2,             None,                   0,   1],
    ['iter',     n_cpus,        None,                   0,   1],
    
    # num_workers through epochs
    ['iter',     0,             None,                   0,   1],
    ['iter',     0,             None,                   0,   2],
    ['iter',     0,             None,                   0,   4],
    ['iter',     0,             None,                   0,   10],
    ['iter',     2,             None,                   0,   1],
    ['iter',     2,             None,                   0,   2],
    ['iter',     2,             None,                   0,   4],
    ['iter',     2,             None,                   0,   10],

    # num_workers with dp
    ['iter',     0,             'dp',                   2,      2],
    ['iter',     1,             'dp',                   2,      2],
    ['iter',     2,             'dp',                   2,      2],
    ['iter',     n_cpus,        'dp',                   2,      2],

    ['iter',     0,             'dp',                   2,      1],
    ['iter',     1,             'dp',                   2,      2],
    ['iter',     2,             'dp',                   2,      4],
    ['iter',     n_cpus,        'dp',                   2,      10],

    # num_workers with ddp
#    ['iter',     0,             'ddp',                  2,      2],
#    ['iter',     1,             'ddp',                  2,      2],
#    ['iter',     2,             'ddp',                  2,      2],
#    ['iter',     n_cpus,        'ddp',                  2,      2],
#
#    ['iter',     0,             'ddp',                  2,      1],
#    ['iter',     1,             'ddp',                  2,      2],
#    ['iter',     2,             'ddp',                  2,      4],
#    ['iter',     n_cpus,        'ddp',                  2,      10],

    # MAP dataset
    # test different num_workers in single node on cpu
    ['map',     0,             None,                   0,   1],
    ['map',     1,             None,                   0,   1],
    ['map',     2,             None,                   0,   1],
    ['map',     n_cpus,        None,                   0,   1],
    
    # num_workers through epochs
    ['map',     0,             None,                   0,   1],
    ['map',     0,             None,                   0,   2],
    ['map',     0,             None,                   0,   4],
    ['map',     0,             None,                   0,   10],
    ['map',     2,             None,                   0,   1],
    ['map',     2,             None,                   0,   2],
    ['map',     2,             None,                   0,   4],
    ['map',     2,             None,                   0,   10],

    # num_workers with dp
   ['map',     0,             'dp',                   2,      2],
   ['map',     1,             'dp',                   2,      2],
   ['map',     2,             'dp',                   2,      2],
   ['map',     n_cpus,        'dp',                   2,      2],

   ['map',     0,             'dp',                   2,      1],
   ['map',     1,             'dp',                   2,      2],
   ['map',     2,             'dp',                   2,      4],
   ['map',     n_cpus,        'dp',                   2,      10],
#
#   # num_workers with ddp
#   ['map',     0,             'ddp',                  2,      2],
#   ['map',     1,             'ddp',                  2,      2],
#   ['map',     2,             'ddp',                  2,      2],
#   ['map',     n_cpus,        'ddp',                  2,      2],
#
#   ['map',     0,             'ddp',                  2,      1],
#   ['map',     1,             'ddp',                  2,      2],
#   ['map',     2,             'ddp',                  2,      4],
#   ['map',     n_cpus,        'ddp',                  2,      10],
])
def test_0(ds_type, num_workers, distributed_backend, gpus, epochs):

    hparams = Namespace(
        batch_size=4,
        val_batch_size=4,
        test_batch_size=4,
        accumulate_grad_batches=3,
        num_workers=num_workers,
        dataset_dir='tests/test_data',
        config_dir='tests/test_data',
        cache_dir='cache',
        output_dir='output',
        max_epochs=epochs,
        max_steps=None,
        max_sequence_length=10,
        gpus=gpus
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
    datamodule = ExampleDataModule(hparams, model, trainer, ds_type=ds_type)
    
    model.datamodule = datamodule
    # Train!
    if datamodule.do_train():
        trainer.fit(model, datamodule=datamodule)

    # Test!
    if datamodule.do_test():
        trainer.test(model, datamodule=datamodule)
