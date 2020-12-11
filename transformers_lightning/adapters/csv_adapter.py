import csv
from typing import Iterable

from transformers_lightning.adapters import FileAdapter
from transformers_lightning.utils import strip_lines


class CSVAdapter(FileAdapter):
    r"""
    An Adapter to load data from a file. Only adds a `filepath` parameter.
    It is still abstract and need to be subclassed.
    """

    def __init__(self, *args, delimiter="\t", quoting=csv.QUOTE_MINIMAL, quotechar='"', **kwargs):
        super().__init__(*args)
        self.csv_kwargs = {'delimiter': delimiter, 'quoting': quoting, 'quotechar': quotechar, **kwargs}

    def __iter__(self) -> Iterable:
        r"""
        Return a generator of parsed lines.
        """
        with open(self.filepath, "r") as fi:
            # use utils.strip_lines to emulate skip_blank_lines of pd.DataFrame
            reader = csv.reader(strip_lines(fi), **self.csv_kwargs)
            yield from reader
