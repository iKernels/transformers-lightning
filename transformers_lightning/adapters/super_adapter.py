import abc
from argparse import Namespace
from typing import Iterable


class SuperAdapter:
    r"""
    The Adapter task is to provide a simple interface to read and prepare a dataset
    for the training phase. An Adapter should read some input data in the correct
    way, pre-process lines if needed (no tokenization) for example by removing
    useless spaces or empty lines. Finally, it should return an iterator over the
    ready-to-be-consumed data. 
    """

    def __init__(self, hparams: Namespace) -> None:
        r"""
        :param hparams: global namespace containing all the useful hyper-parameters
        """
        assert isinstance(hparams, Namespace), f"Argument `hparams` must be of type `Namespace`"
        self.hparams = hparams

    @abc.abstractmethod
    def __iter__(self) -> Iterable:
        r"""
        This function should use the arguments in `hparams` to read the file
        from the disk and return an iterator over the (parsed) lines.
        This is the right place to parse csv files and yield each parsed line for example.

        >>> with open(self.hparams.filepath, "r") as fi:
        >>>     for line in csv.reader(fi):
        >>>         yield line
        """

    @abc.abstractmethod
    def preprocess_line(self, line: list) -> list:
        r"""
        Process a line. The structure of each line is exactly
        the same returned by the __iter__ method. Here you should do data preparation
        for the actual model being trained. This is a good place to do batch tokenization,
        padding and so on.

        >>> sentences = line[0]
        >>> results = self.hparams.tokenizer.encode_plus(sentences,
                                                         truncation=True,
                                                         add_special_tokens=True,
                                                         padding='max_length',
                                                         max_length=self.hparams.max_sequence_length)
        >>> return results
        """