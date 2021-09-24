from argparse import Namespace

from pytorch_lightning import Trainer
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModelWithOptim, get_random_gpus_list, standard_args


def do_test_fix_max_steps(max_epochs, accumulate_grad_batches, batch_size, expected_max_steps, **kwargs):

    hyperparameters = Namespace(
        batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=2,
        max_epochs=max_epochs,
        max_steps=None,
        iterable=False,
        **standard_args,
        **kwargs
    )

    if hasattr(hyperparameters, "gpus"):
        hyperparameters.gpus = get_random_gpus_list(hyperparameters.gpus)

    # instantiate PL trainer
    trainer = Trainer.from_argparse_args(hyperparameters)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # not checking ids because sometimes the sampler will duplicate elements to fill all gpus
    model = DummyTransformerModelWithOptim(hyperparameters)

    # Datasets
    datamodule = DummyDataModule(hyperparameters, train_number=3, test_number=3, valid_number=3, tokenizer=tokenizer)
    trainer.fit(model, datamodule=datamodule)

    # Assert max steps computed correctly
    assert model.num_training_steps == expected_max_steps
