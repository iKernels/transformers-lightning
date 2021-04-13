from argparse import Namespace
import torch

from transformers import BertConfig, BertForSequenceClassification, AdamW

from transformers_lightning.adapters import CSVAdapter, TransformersAdapter
from transformers_lightning.datamodules import AdaptersDataModule
from transformers_lightning.models import TransformersModel

standard_args = dict(
    output_dir='/tmp/output',
    max_sequence_length=10,
    learning_rate=1e-04,
    adam_epsilon=1e-07,
    adam_betas=[0.9, 0.99],
    warmup_steps=0,
    beg_scheduler_step=0,
    weight_decay=0.1,
    padding='max_length',
)


# Adapters
class DummyCSVAdapter(CSVAdapter):
    """ Only preprocess a line by splitting and doing types conversion. """

    def preprocess_line(self, line: list) -> list:
        return [int(line[0]), int(line[1]), int(line[2]), line[3], line[4], eval(line[5])]


class DummyTransformersAdapter(TransformersAdapter):
    """ Tokenizer a sentence and compute word tails. """

    def preprocess_line(self, line: list) -> list:
        results = self.tokenizer.encode_plus(line[3], line[4], padding=self.hparams.padding)
        results['words_tails'] = self._convert_ids_to_word_tails(results['input_ids'])
        results['ids'] = int(line[0])
        results['labels'] = (line[5].lower().strip() == "true")
        return results


# DataModules
class DummyDataModule(AdaptersDataModule):

    def __init__(self, hparams, test_number=1, tokenizer=None):
        super().__init__(hparams)
        self.train_adapter = DummyTransformersAdapter(
            self.hparams, f"tests/data/test{test_number}.tsv", delimiter="\t", tokenizer=tokenizer
        )
        self.valid_adapter = DummyTransformersAdapter(
            self.hparams, f"tests/data/test{test_number}.tsv", delimiter="\t", tokenizer=tokenizer
        )
        self.test_adapter = [
            DummyTransformersAdapter(self.hparams, f"tests/data/test{test_number}.tsv", delimiter="\t", tokenizer=tokenizer)
            for _ in range(2)
        ]


# Models
class DummyTransformerModel(TransformersModel):

    def __init__(self, hparams):
        super().__init__(hparams)

        # super light BERT model
        config = BertConfig(hidden_size=12, num_hidden_layers=1, num_attention_heads=1, intermediate_size=12)
        self.model = BertForSequenceClassification(config)

    def training_step(self, batch, batch_idx):
        """ Training step on BertForSequenceClassification. """
        batch['labels'] = batch['labels'].to(dtype=torch.long)
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs)
        return {'loss': results.loss, 'ids': batch['ids']}

    def training_step_end(self, batch_parts):
        batch_parts['loss'] = torch.sum(batch_parts['loss'])
        return batch_parts

    def training_epoch_end(self, outputs):
        ids = torch.cat([o['ids'] for o in outputs], dim=0)
        # in distributed mode collect ids from every process (gpu)
        if self.trainer.distributed_backend in ["ddp", "ddp_cpu"]:
            gather_ids = [torch.zeros_like(ids) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gather_ids, ids)
            ids = torch.cat(gather_ids, dim=0)

        received = torch.zeros(len(self.datamodule.train_dataset)).to(dtype=bool)
        received[ids] = True

        # assert no duplicate element received
        assert len(set(ids.tolist())) == len(
            ids.tolist()
        ), (f"Received {len(ids.tolist())} ids but only {len(set(ids.tolist()))} are unique: {ids}")
        # assert all elements received
        assert all(received), (f"({self.trainer.max_steps}) Received not all {len(received)} ids: {received}")

    def validation_step(self, batch, batch_idx):
        batch['labels'] = batch['labels'].to(dtype=torch.long)
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs)
        return results.loss

    def test_step(self, batch, batch_idx, dataset_idx):
        batch['labels'] = batch['labels'].to(dtype=torch.long)
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs)
        return results.loss
