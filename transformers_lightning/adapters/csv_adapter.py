import csv
from typing import Iterable

from transformers_lightning import utils
from transformers_lightning.adapters import FileAdapter
from transformers_lightning.utils import strip_lines


class CSVAdapter(FileAdapter):

    def __init__(self,
        *args,
        delimiter=",",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"'
    ):
        super().__init__(*args)
        self.delimiter = delimiter
        self.quoting = quoting
        self.quotechar = quotechar

    def __iter__(self) -> Iterable:
        """ Return a generator of parsed lines. """
        with open(self.filepath, "r") as fi:
            # use utils.strip_lines to emulate skip_blank_lines of pd.DataFrame
            reader = csv.reader(
                strip_lines(fi),
                delimiter=self.delimiter,
                quoting=self.quoting,
                quotechar=self.quotechar
            )
            yield from reader
