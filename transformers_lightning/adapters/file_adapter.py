import os
from argparse import Namespace
from transformers_lightning.adapters.super_adapter import SuperAdapter


class FileAdapter(SuperAdapter):
    """
    An Adapter to load data from a file. Only adds a `filepath` parameter.
    It is still abstract and need to be subclassed.
    """

    def __init__(self, hparams: Namespace, filepath: str) -> None:
        """
        :param filepath: must be relative to the `dataset_dir` defined in hparams
        """
        super().__init__(hparams)
        
        assert isinstance(filepath, str), f"Argument `filepath` must be of type `str`"
        self.filepath = os.path.join(hparams.dataset_dir, filepath)
