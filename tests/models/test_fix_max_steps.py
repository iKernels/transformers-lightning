import multiprocessing
import time
from argparse import Namespace

import pytest
import pytorch_lightning as pl
import torch
from transformers import AdamW, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertForSequenceClassification

from tests import adapters
from transformers_lightning import adapters, datamodules, models, utils
from transformers_lightning.adapters import CSVAdapter
from transformers_lightning.datamodules import AdaptersDataModule
from transformers_lightning.models import TransformersModel

n_cpus = multiprocessing.cpu_count()


class SimpleTransformerLikeModel(TransformersModel):

    def __init__(self, hparams):
        super().__init__(hparams)

        # super light BERT model
        config = BertConfig(hidden_size=12, num_hidden_layers=1, num_attention_heads=1, intermediate_size=12)
        self.model = BertForSequenceClassification(config)

    def configure_optimizers(self):
        self.computed_max_steps = utils.compute_max_steps(self.hparams, self.trainer)
        return AdamW(self.model.parameters())

    def training_step(self, batch, batch_idx):
        batch['labels'] = batch['labels'].to(dtype=torch.long)
        results = self(**batch)
        return results.loss


class ExampleAdapter(CSVAdapter):

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def preprocess_line(self, line: list) -> list:

        results = self.tokenizer.encode_plus(
            line[3],
            line[4],
            add_special_tokens=True,
            padding='max_length',
            max_length=self.hparams.max_sequence_length,
            truncation=True
        )
        res = {**results, 'labels': line[5].strip().lower() == "true"}
        return res


class ExampleDataModule(AdaptersDataModule):

    def __init__(self, *args, test_number=1, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_adapter = ExampleAdapter(
            self.hparams, f"tests/test_data/test{test_number}.tsv", delimiter="\t", tokenizer=tokenizer
        )


# Test if max_steps fix works correctly
@pytest.mark.parametrize(
    ["max_epochs", "accumulate_grad_batches", "batch_size", "expected_max_steps"], [
        [1, 1, 4, 10],
        [1, 3, 8, 2],
        [4, 2, 12, 8],
        [4, 4, 16, 4],
        [1, 1, 4, 10],
        [1, 3, 8, 2],
        [4, 4, 16, 4],
        [4, 2, 12, 8],
    ]
)
def test_fix_max_steps_cpu(max_epochs, accumulate_grad_batches, batch_size, expected_max_steps):

    hparams = Namespace(
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=4,
        output_dir='output',
        max_epochs=max_epochs,
        max_steps=None,
        max_sequence_length=10,
        gpus=None,
        skip_in_training=None
    )

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        profiler='simple',
        logger=None,
        callbacks=[],
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # instantiate PL model
    model = SimpleTransformerLikeModel(hparams)

    # Datasets
    datamodule = ExampleDataModule(hparams, tokenizer=tokenizer)

    # Train!
    trainer.fit(model, datamodule=datamodule)
    assert model.computed_max_steps == expected_max_steps


# Test if max_steps fix works correctly
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.parametrize(
    ["max_epochs", "accumulate_grad_batches", "batch_size", "distributed_backend", "expected_max_steps"], [
        [1, 1, 4, 'dp', 10],
        [1, 3, 8, 'dp', 2],
        [4, 2, 12, 'dp', 8],
        [4, 4, 16, 'dp', 4],
        [1, 1, 4, 'ddp', 5],
        [1, 3, 8, 'ddp', 1],
        [4, 2, 12, 'ddp', 4],
        [4, 4, 16, 'ddp', 4],
    ]
)
def test_fix_max_steps_gpu(max_epochs, accumulate_grad_batches, batch_size, distributed_backend, expected_max_steps):

    time.sleep(5)

    hparams = Namespace(
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=4,
        output_dir='output',
        max_epochs=max_epochs,
        max_steps=None,
        max_sequence_length=10,
        gpus=2,
        skip_in_training=None,
        accelerator=distributed_backend
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

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
    datamodule = ExampleDataModule(hparams, tokenizer=tokenizer)

    # Train!
    trainer.fit(model, datamodule=datamodule)

    assert model.computed_max_steps == expected_max_steps
