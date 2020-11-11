import multiprocessing
from argparse import Namespace
from transformers_lightning.datasets.iterable_dataset import TransformersIterableDataset
import os
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch import tensor
from torch.utils.data.dataset import IterableDataset
import transformers_lightning
from transformers import BertTokenizer
from transformers_lightning import utils
from transformers.modeling_bert import (BertConfig,
                                        BertForSequenceClassification)

n_cpus = multiprocessing.cpu_count()
N = 20

class SimpleTransformerLikeModel(transformers_lightning.models.SuperModel):

    def __init__(self, hparams):
        super().__init__(hparams)

        # super light BERT model
        self.lin = torch.nn.Linear(10, 1)

    def training_step(self, batch, batch_idx):
        return { 'loss': self.lin(batch["input_ids"].to(dtype=torch.float32)).mean(), 'ids': batch['ids'] }

    def training_epoch_end(self, outputs):
        ids = torch.cat([o['ids'] for o in outputs], dim=0)

        print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} returned ids: {ids}")
        # in distributed mode collect ids from every process (gpu)
        if torch.distributed.is_initialized():
            gather_ids = [torch.ones_like(ids) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gather_ids, ids)
            
            ids = torch.cat([x.to(ids) for x  in gather_ids], dim=0)
            print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} ALL ids: {ids}")
        try:
            received = torch.zeros((len(self.datamodule.train_dataset),)).to(dtype=bool)
        except TypeError:
            received = torch.zeros((self.datamodule.train_dataset.length,)).to(dtype=bool)
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

"""
class ExampleDataModule(transformers_lightning.datamodules.SuperDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
 
    train_dataloader = transformers_lightning.datamodules.SuperDataModule.default_train_dataloader
"""


class ExampleDataModule(pl.LightningDataModule):

    def get_config(self, config_file):
        """ Load a config file from standard directory and check that file exists! """
        config_path = os.path.join(self.hparams.config_dir, "datasets", config_file)
        assert os.path.isfile(config_path), f"Specified config {config_path} does not exist!"
        return utils.load_yaml(config_path)

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.train_config = self.get_config("dataset.yaml")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def setup(self, stage=None):
        self.train_dataset = TransformersIterableDataset(
            self.hparams, self.tokenizer, self.train_config
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=utils.collate_single_fn)

hparams = Namespace(
    batch_size=4,
    val_batch_size=4,
    test_batch_size=4,
    accumulate_grad_batches=3,
    num_workers=2,
    dataset_dir='tests/test_data',
    config_dir='tests/test_data',
    cache_dir='cache',
    output_dir='output',
    max_epochs=2,
    max_steps=None,
    max_sequence_length=10,
    gpus=2,
    distributed_backend='ddp',
    dataset_style='iter'
)

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

model.datamodule = datamodule
trainer.fit(model, datamodule=datamodule)
