import multiprocessing
from argparse import Namespace

import pytest
import pytorch_lightning as pl
import torch
from transformers_lightning import models, datamodules
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer
from transformers.modeling_bert import (BertConfig,
                                        BertForSequenceClassification)

n_cpus = multiprocessing.cpu_count()

class SimpleTransformerLikeModel(models.SuperModel):

    def __init__(self, hparams):
        super().__init__(hparams)

        # super light BERT model
        config = BertConfig(hidden_size=12, num_hidden_layers=1, num_attention_heads=1, intermediate_size=12)
        self.model = BertForSequenceClassification(config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", config=config, cache_dir=hparams.cache_dir)

    def training_step(self, batch, batch_idx):
        print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} processing ids: {batch['ids']}")
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return { 'loss': results.loss, 'ids': batch['ids'] }

        """    def training_step_end(self, batch_parts):
        batch_parts['loss'] = torch.sum(batch_parts['loss'])
        return batch_parts"""

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
            expected_len = (
                self.datamodule.train_dataset.length // torch.distributed.get_world_size()
            ) * torch.distributed.get_world_size()
            received = torch.zeros((expected_len,)).to(dtype=bool)
        received[ids] = True

        # assert no duplicate element received
        assert len(set(ids.tolist())) == len(ids.tolist()), (
            f"Received {len(ids.tolist())} ids but only {len(set(ids.tolist()))} are unique: {ids}"
        )
        # assert all elements received
        assert all(received), (
            f"({self.trainer.max_steps}) Received not all {len(received)} ids: {received}"
        )

    def validation_step(self, batch, batch_idx):
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return results.loss

    def test_step(self, batch, batch_idx):
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return results.loss


class ExampleDataModule(datamodules.SuperDataModule):

    def __init__(self, *args, train_config=None, **kwargs):
        super().__init__(*args, **kwargs)

        if train_config is None:
            self.train_config = "dataset1.yaml"
        else:
            self.train_config = train_config
 
    train_dataloader = datamodules.SuperDataModule.default_train_dataloader



    # ITER dataset
    # num_workers with ddp
test = [
    ['iter',     0,             'ddp',                  2,      1],
    ['iter',     1,             'ddp',                  2,      2],
    ['iter',     2,             'ddp',                  2,      2],
    ['iter',     0,             'ddp',                  2,      1],
    ['iter',     n_cpus,        'ddp',                  2,      10],
]

def test_datamodule_gpu_ddp_only(ds_type, num_workers, distributed_backend, gpus, epochs):

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
        gpus=gpus,
        dataset_style=ds_type,
        distributed_backend=distributed_backend
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



for t in test:
    test_datamodule_gpu_ddp_only(*t)
 