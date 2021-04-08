import os
from argparse import Namespace

from transformers_lightning.adapters.super_adapter import SuperAdapter


class FileAdapter(SuperAdapter):
    r"""
    An Adapter to load data from a file. Only adds a `filepath` parameter.
    It is still abstract and need to be subclassed.
    """

    def __init__(self, hparams: Namespace, filepath: str) -> None:
        r"""
        Args:
            filepath: path of the file that should be loaded
        """
        super().__init__(hparams)

        assert isinstance(filepath, str), (f"Argument `filepath` must be of type `str`")
        assert os.path.isfile(filepath), (f"{filepath} is not a correct path to a file")
        self.filepath = filepath
