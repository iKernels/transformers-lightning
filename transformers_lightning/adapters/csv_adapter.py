import csv
from typing import Iterable

from transformers_lightning import utils
from transformers_lightning.adapters import FileAdapter


class CSVAdapter(FileAdapter):

    @staticmethod
    def convert_line(line: list) -> list:
        """
        Convert strings to integer or floats if possible
        """
        res = []
        for x in line:
            try:
                res.append(eval(x))
            except:
                res.append(x)
        return res

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
                utils.strip_lines(fi),
                delimiter=self.delimiter,
                quoting=self.quoting,
                quotechar=self.quotechar
            )
            for line in reader:
                yield CSVAdapter.convert_line(line)
