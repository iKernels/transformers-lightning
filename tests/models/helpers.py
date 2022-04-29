from argparse import Namespace

from pytorch_lightning import Trainer
from transformers import BertTokenizer

from tests.helpers import DummyDataModule, DummyTransformerModelWithOptim, standard_args


def do_test_fix_max_steps(max_epochs, accumulate_grad_batches, batch_size, **kwargs):

    hyperparameters = Namespace(
        batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        num_workers=2,
        max_epochs=max_epochs,
        max_steps=-1,
        iterable=False,
        **standard_args,
        **kwargs
    )

    # instantiate PL trainer
    trainer = Trainer.from_argparse_args(hyperparameters)

    tokenizer = BertTokenizer('tests/data/vocab.txt')
    # not checking ids because sometimes the sampler will duplicate elements to fill all gpus
    model = DummyTransformerModelWithOptim(hyperparameters)

    # Datasets
    datamodule = DummyDataModule(hyperparameters, length_train=40, length_test=40, length_valid=40, tokenizer=tokenizer)
    trainer.fit(model, datamodule=datamodule)

    compare_fn = kwargs['compare_fn'] if 'compare_fn' in kwargs else lambda a, b: a == b
    assert compare_fn(trainer.global_step, model.computed_steps), (
        f"global {trainer.global_step} steps but computed {model.computed_steps}"
    )
