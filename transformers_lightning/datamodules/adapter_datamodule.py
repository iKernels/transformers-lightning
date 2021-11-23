from argparse import Namespace
from typing import Callable, Union

from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.trainer import Trainer

from transformers_lightning.adapters.super_adapter import SuperAdapter
from transformers_lightning.datamodules.super_datamodule import SuperDataModule
from transformers_lightning.datasets.iterable_dataset import TransformersIterableDataset
from transformers_lightning.datasets.map_dataset import TransformersMapDataset
from transformers_lightning.utils.functional import collate_single_fn


class AdaptersDataModule(SuperDataModule):
    r"""
    AdaptersDataModule should be used when you want to read and tokenizer data on-the-fly.
    It implements some simple methods to check whether training, val or testing is required.
    It work with adapters: you could define them inside the `__init__()` method or pass them
    as arguments.
    """

    def __init__(
        self,
        hyperparameters: Namespace,
        trainer: Trainer,
        collate_fn: Callable = collate_single_fn,
        train_adapter: SuperAdapter = None,
        valid_adapter: SuperAdapter = None,
        test_adapter: SuperAdapter = None,
        predict_adapter: SuperAdapter = None,
    ):
        super().__init__(hyperparameters, trainer, collate_fn)

        # instantiate eventual adapters passed from init method
        if train_adapter is not None:
            assert isinstance(train_adapter, SuperAdapter), "Argument `train_adapter` must be of type `SuperAdapter`"
            self.train_adapter = train_adapter

        if valid_adapter is not None:
            assert isinstance(valid_adapter, SuperAdapter), "Argument `valid_adapter` must be of type `SuperAdapter`"
            self.valid_adapter = valid_adapter

        if test_adapter is not None:
            assert (
                isinstance(test_adapter, SuperAdapter) or isinstance(test_adapter, list)
            ), "Argument `test_adapter` must be of type `SuperAdapter` or List[SuperAdapter]"

            if isinstance(test_adapter, list):
                for adapter in test_adapter:
                    assert isinstance(
                        adapter, SuperAdapter
                    ), "Argument `test_adapter` must be of type `SuperAdapter` or List[SuperAdapter]"
            self.test_adapter = test_adapter

        if predict_adapter is not None:
            assert isinstance(
                predict_adapter, SuperAdapter
            ), "Argument `predict_adapter` must be of type `SuperAdapter`"
            self.predict_adapter = predict_adapter

        r"""
        This space should be used to instantiate the Adapters it they were not passed through the kwargs

        >>> self.train_adapter = CSVAdapter(self.hyperparameters, "pre-training/train.tsv", delimiter="\t")
        >>> self.valid_adapter = CSVAdapter(self.hyperparameters, "pre-training/valid.tsv", delimiter="\t")
        >>> self.test_adapter = CSVAdapter(self.hyperparameters, "pre-training/test.tsv", delimiter="\t")
        >>> self.predict_adapter = CSVAdapter(self.hyperparameters, "pre-training/predict.tsv", delimiter="\t")
        """

    def get_dataset(self, adapter: SuperAdapter) -> Union[TransformersMapDataset, TransformersIterableDataset]:
        r""" Return iterable or map dataset from adapter. """
        if self.hyperparameters.iterable:
            return TransformersIterableDataset(self.hyperparameters, adapter, self.trainer)
        else:
            return TransformersMapDataset(self.hyperparameters, adapter, self.trainer)

    # Optional, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        r"""
        Load datasets only if respective Adapter are defined.
        This implementation should be enough for most subclasses.
        """
        if stage == TrainerFn.FITTING.value or stage == TrainerFn.VALIDATING.value:
            if self.do_train():
                self.train_dataset = self.get_dataset(self.train_adapter)
            if self.do_validation():
                self.valid_dataset = self.get_dataset(self.valid_adapter)

        elif stage == TrainerFn.TESTING.value:
            if self.do_test():
                if isinstance(self.test_adapter, SuperAdapter):
                    self.test_adapter = [self.test_adapter]
                self.test_dataset = [
                    self.get_dataset(adapter) for adapter in self.test_adapter
                ]

        elif stage == TrainerFn.PREDICTING.value:
            if self.do_predict():
                self.predict_adapter = self.get_dataset(self.predict_adapter)

    def do_train(self):
        return self.train_adapter is not None

    def do_validation(self):
        return self.valid_adapter is not None

    def do_test(self):
        return self.test_adapter is not None

    def do_predict(self):
        return self.predict_adapter is not None
