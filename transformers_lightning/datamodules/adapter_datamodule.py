from argparse import Namespace

from transformers_lightning.adapters.super_adapter import SuperAdapter
from transformers_lightning.datamodules.super_datamodule import SuperDataModule
from transformers_lightning.datasets.iterable_dataset import TransformersIterableDataset
from transformers_lightning.datasets.map_dataset import MapDataset


class AdaptersDataModule(SuperDataModule):
    """
    AdaptersDataModule should be used when you want to read and tokenizer data on-the-fly.
    It implements some simple methods to check whether training, val or testing is required.
    It work with adapters: you could define them inside the `__init__()` method or pass them
    as arguments.
    """

    def __init__(
        self,
        hyperparameters: Namespace,
        train_adapter: SuperAdapter = None,
        valid_adapter: SuperAdapter = None,
        test_adapter: SuperAdapter = None,
        predict_adapter: SuperAdapter = None,
    ):
        super().__init__(hyperparameters)

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
        """
        This space should be used to instantiate the Adapters it they were not passed through the kwargs

        >>> self.train_adapter = CSVAdapter(self.hyperparameters, "pre-training/train.tsv", delimiter="\t")
        >>> self.valid_adapter = CSVAdapter(self.hyperparameters, "pre-training/valid.tsv", delimiter="\t")
        >>> self.test_adapter = CSVAdapter(self.hyperparameters, "pre-training/test.tsv", delimiter="\t")
        >>> self.predict_adapter = CSVAdapter(self.hyperparameters, "pre-training/predict.tsv", delimiter="\t")
        """

    # Optional, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        """
        Load datasets only if respective Adapter are defined.
        This implementation should be enough for most subclasses.
        """
        dataset_class = MapDataset if not self.hyperparameters.iterable else TransformersIterableDataset

        if stage == 'fit':
            if self.train_adapter is not None:
                self.train_dataset = dataset_class(self.hyperparameters, self.train_adapter, self.trainer)
            if self.valid_adapter is not None:
                self.valid_dataset = dataset_class(self.hyperparameters, self.valid_adapter, self.trainer)

        elif stage == 'test':
            if self.test_adapter is not None:
                if isinstance(self.test_adapter, SuperAdapter):
                    self.test_adapter = [self.test_adapter]
                self.test_dataset = [
                    dataset_class(self.hyperparameters, adapter, self.trainer) for adapter in self.test_adapter
                ]

        elif stage == 'predict':
            if self.predict_dataset is not None:
                self.predict_adapter = dataset_class(self.hyperparameters, self.predict_adapter, self.trainer)

    def do_train(self):
        return self.train_adapter is not None

    def do_validation(self):
        return self.valid_adapter is not None

    def do_test(self):
        return self.test_adapter is not None

    def do_predict(self):
        return self.predict_adapter is not None
