# Adapters

An `Adapter` is a standardized interface to read data from disk. Each `Adapter` can be called in the same way, regarding the structure of the dataset that is being read.

An `Adapter` must override at least two methods of the superclass `SuperAdapter`: `__iter__` and `preprocess_line`. The first method should return a simple iterator over all the data in the dataset. The second, that will likely be called in a multiprocessing environment, should do the necessary preprocessing and prepare the data for the neural network. Please move the heavy work, like tokenization, to the `preprocess_line` method because this will be parallelized over many CPU cores. `__iter__` is usually only a function to read from the disk and yield data line by line.

The following example illustrates how to build an `Adapter` to read data from a `tsv` file and how to implement tokenization in the `preprocess_line` method.

The `__iter__` method does not has arguments other than `self`. Paramters should be retrieved through `self.hparams`.
`preprocess_line` instead receives a line at a time and is strongly recommended to return a `dict`. `dict` improves readability and values are automagically concatenated by the `SuperDataModule` class.

```python
import csv
from transformers_lightning import SuperAdapter


class TransformersAdapter(SuperAdapter):

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterable:
        """ Return a generator of parsed lines. """
        with open(self.filepath, "r") as fi:
            reader = csv.reader(fi, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            yield from reader

    def preprocess_line(self, line: list) -> list:
        """
        Tokenize a single line. Suppose each line
        containt two sentences and a label.
        """
        results = self.tokenizer.encode_plus(
            line[0], line[1],
            padding='max_length',
            max_length=self.hparams.max_sequence_length,
            truncation=True
        )
        res = { **results, 'label': line[2] }
        return res
```