import multiprocessing

import torch
from transformers_lightning import models, datamodules
from transformers import AdamW, BertTokenizer
from transformers.modeling_bert import (BertConfig,
                                        BertForSequenceClassification)

n_cpus = multiprocessing.cpu_count()

class SimpleTransformerLikeModel(models.SuperModel):

    def __init__(self, hparams, do_ids_check=True):
        super().__init__(hparams)
        self.do_ids_check = do_ids_check

        # super light BERT model
        config = BertConfig(hidden_size=12, num_hidden_layers=1, num_attention_heads=1, intermediate_size=12)
        self.model = BertForSequenceClassification(config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", config=config, cache_dir=hparams.cache_dir)

    def training_step(self, batch, batch_idx):
        if self.trainer.distributed_backend == "ddp":
            print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} processing ids: {batch['ids']}")
 
        kwargs = {k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        results = self(**kwargs, return_dict=True)
        return { 'loss': results.loss, 'ids': batch['ids'] }

    def training_step_end(self, batch_parts):
        batch_parts['loss'] = torch.sum(batch_parts['loss'])
        return batch_parts

    def training_epoch_end(self, outputs):
        ids = torch.cat([o['ids'] for o in outputs], dim=0)

        # in distributed mode collect ids from every process (gpu)
        if self.trainer.distributed_backend == "ddp":
            print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} returned ids: {ids}")

            gather_ids = [torch.ones_like(ids) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gather_ids, ids)
            print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} gather ids: {gather_ids}")

            ids = torch.cat(gather_ids, dim=0)
            print(f"ID {torch.distributed.get_rank()}/{torch.distributed.get_world_size()} ALL ids: {ids}")

        try:
            received = torch.zeros((len(self.datamodule.train_dataset),)).to(dtype=bool)
        except TypeError:
            if self.trainer.distributed_backend == "ddp":
                expected_len = (
                    self.datamodule.train_dataset.length // torch.distributed.get_world_size()
                ) * torch.distributed.get_world_size()
            else:
                expected_len = self.datamodule.train_dataset.length
            received = torch.zeros((expected_len,)).to(dtype=bool)
        received[ids] = True

        if self.do_ids_check:
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
