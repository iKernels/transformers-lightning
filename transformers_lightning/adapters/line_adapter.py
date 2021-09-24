from argparse import Namespace
from typing import Iterable

from transformers_lightning.adapters.file_adapter import FileAdapter
from transformers_lightning.utils import strip_lines


class LineAdapter(FileAdapter):
    r"""
    An Adapter to load data from a text file, line by line. Only adds a `filepath` parameter.
    """

    def __init__(self, hyperparameters: Namespace, filepath: str):
        super().__init__(hyperparameters, filepath)

    def __iter__(self) -> Iterable:
        r"""
        Return a generator of parsed lines.
        """
        with open(self.filepath, "r") as fi:
            # use utils.strip_lines to emulate skip_blank_lines of pd.DataFrame
            yield from strip_lines(fi)
