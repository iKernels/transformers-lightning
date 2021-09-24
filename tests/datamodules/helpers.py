from argparse import Namespace

import pytorch_lightning as pl
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModelWithOptim, get_random_gpus_list, standard_args


def do_test_datamodule(num_workers, batch_size, accumulate_grad_batches, accelerator, iterable, **kwargs):

    hyperparameters = Namespace(
        batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=num_workers,
        max_epochs=1,
        max_steps=None,
        accelerator=accelerator,
        iterable=iterable,
        **standard_args,
        **kwargs,
    )

    if hasattr(hyperparameters, "gpus"):
        hyperparameters.gpus = get_random_gpus_list(hyperparameters.gpus)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # instantiate PL trainer and model
    trainer = pl.Trainer.from_argparse_args(hyperparameters)
    model = DummyTransformerModelWithOptim(hyperparameters, check_ids=True)

    # Datasets
    datamodule = DummyDataModule(hyperparameters, train_number=2, valid_number=2, test_number=2, tokenizer=tokenizer)
    model.datamodule = datamodule
    trainer.fit(model, datamodule=datamodule)
