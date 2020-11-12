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
        print(f"Training {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} got ids: {batch['ids']}")
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return { 'loss': results.loss, 'ids': batch['ids'] }

    def training_epoch_end(self, outputs):
        ids = torch.cat([o['ids'] for o in outputs], dim=0)

        print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} returned ids: {ids}")
        # in distributed mode collect ids from every process (gpu)
        if self.trainer.distributed_backend == "ddp":
            gather_ids = [torch.ones_like(ids) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gather_ids, ids)
            print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} gather ids: {gather_ids}")

            ids = torch.cat(gather_ids, dim=0)
            print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} ALL ids: {ids}")

        try:
            received = torch.zeros((len(self.datamodule.train_dataset),)).to(dtype=bool)
        except TypeError:
            received = torch.zeros(
                (self.datamodule.train_dataset.limit if self.datamodule.train_dataset.limit is not None else self.datamodule.train_dataset.length,)
            ).to(dtype=bool)
        received[ids] = True

        # assert no duplicate element received
        print(len(set(ids.tolist())) == len(ids.tolist()), (
            f"Received {len(ids.tolist())} ids but only {len(set(ids.tolist()))} are unique: {ids}"
        ))
        # assert all elements received
        print(all(received), (
            f"({self.trainer.max_steps}) Received not all {len(received)} ids: {received}"
        ))


    def validation_step(self, batch, batch_idx):
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return results.loss

    def test_step(self, batch, batch_idx):
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return results.loss


class ExampleDataModule(transformers_lightning.datamodules.SuperDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_config = "dataset3.yaml"
 
    train_dataloader = transformers_lightning.datamodules.SuperDataModule.default_train_dataloader


hparams = Namespace(
    batch_size=4,
    val_batch_size=4,
    test_batch_size=4,
    accumulate_grad_batches=3,
    num_workers=1,
    dataset_dir='tests/test_data',
    config_dir='tests/test_data',
    cache_dir='cache',
    output_dir='output',
    max_epochs=1,
    max_steps=None,
    max_sequence_length=10,
    gpus=2,
    dataset_style='iter',
    distributed_backend='ddp'
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
datamodule = ExampleDataModule(hparams, model, trainer)

model.datamodule = datamodule
trainer.fit(model, datamodule=datamodule)