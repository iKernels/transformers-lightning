import random
from argparse import Namespace
from typing import Iterable

import torch
from pytorch_lightning.utilities.data import has_len
from pytorch_lightning.utilities.distributed import distributed_available
from transformers import BertConfig, BertForSequenceClassification
from transformers.optimization import AdamW
from transformers.tokenization_utils import PreTrainedTokenizerBase

from transformers_lightning.adapters import SuperAdapter
from transformers_lightning.datamodules import AdaptersDataModule
from transformers_lightning.language_modeling.utils import whole_word_tails_mask
from transformers_lightning.models import TransformersModel

standard_args = dict(
    output_dir='/tmp/output',
    learning_rate=1e-04,
    adam_epsilon=1e-07,
    adam_betas=[0.9, 0.99],
    warmup_steps=0,
    beg_scheduler_step=0,
    weight_decay=0.1,
    padding='max_length',
    max_length=5,
)


def get_random_gpus_list(number_of_gpus):

    if (not torch.cuda.is_available()) or (number_of_gpus is None) or (number_of_gpus == 0):
        return None

    gpus_ids = random.sample(range(torch.cuda.device_count()), k=number_of_gpus)
    return gpus_ids


class DummyTransformersAdapter(SuperAdapter):
    """ Tokenizer a sentence and compute word tails. """

    def __init__(self, hyperparameters: Namespace, length: int, tokenizer: PreTrainedTokenizerBase):
        super().__init__(hyperparameters)
        self.length = length
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterable:
        r"""
        Return a generator of parsed lines.
        """
        for i in range(self.length):
            yield (
                i,
                random.randint(0, 10000),
                random.randint(0, 10000),
                f"This is a question {i}",
                f"This is an answer {i}",
                False
            )

    def preprocess_line(self, line: list) -> list:
        results = self.tokenizer(
            line[3],
            line[4],
            padding=self.hyperparameters.padding,
            max_length=self.hyperparameters.max_length,
            truncation=True,
        )
        results['words_tails'] = whole_word_tails_mask(results['input_ids'], tokenizer=self.tokenizer)
        results['ids'] = line[0]
        results['labels'] = line[5]
        return results


# DataModules
class DummyDataModule(AdaptersDataModule):

    def __init__(
        self,
        hyperparameters,
        trainer=None,
        length_train=17,
        length_valid=96,
        length_test=40,
        tokenizer=None
    ):
        super().__init__(hyperparameters, trainer)
        self.train_adapter = DummyTransformersAdapter(
            self.hyperparameters, length=length_train, tokenizer=tokenizer
        )
        self.valid_adapter = DummyTransformersAdapter(
            self.hyperparameters, length=length_valid, tokenizer=tokenizer
        )
        self.test_adapter = [
            DummyTransformersAdapter(
                self.hyperparameters, length=length_test, tokenizer=tokenizer
            ) for _ in range(2)
        ]


# Models
class DummyTransformerModel(TransformersModel):

    def __init__(self, hyperparameters, check_ids: bool = False):
        super().__init__(hyperparameters)
        self.check_ids = check_ids

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
        if distributed_available():
            ids = self.all_gather(ids).view(-1)

        if has_len(self.trainer.datamodule.train_dataset):
            received = torch.zeros(len(self.trainer.datamodule.train_dataset)).to(dtype=bool)
        else:
            received = torch.zeros(len(list(self.trainer.datamodule.train_dataset))).to(dtype=bool)
        received[ids] = True

        if self.check_ids:
            # assert no duplicate element received
            assert len(set(ids.tolist())) == len(
                ids.tolist()
            ), (f"Received {len(ids.tolist())} ids but only"
                f" {len(set(ids.tolist()))} are unique: {ids}")
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


class DummyTransformerModelWithOptim(DummyTransformerModel):

    def configure_optimizers(self):
        self.computed_steps = self.num_training_steps()
        return AdamW(self.model.parameters())
