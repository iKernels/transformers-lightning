from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Iterable


class SuperAdapter(ABC):
    r"""
    The Adapter task is to provide a simple interface to read and prepare a dataset
    for the training phase. An Adapter should read some input data in the correct
    way, pre-process lines if needed (no tokenization) for example by removing
    useless spaces or empty lines. Finally, it should return an iterator over the
    ready-to-be-consumed data.
    """

    def __init__(self, hyperparameters: Namespace) -> None:
        r"""
        :param hyperparameters: global namespace containing all the useful hyper-parameters
        """
        self.hyperparameters = hyperparameters

    @abstractmethod
    def __iter__(self) -> Iterable:
        r"""
        This function should use the arguments in `hyperparameters` to read the file
        from the disk and return an iterator over the (parsed) lines.
        This is the right place to parse csv files and yield each parsed line for example.

        >>> with open(self.hyperparameters.filepath, "r") as fi:
        >>>     for line in csv.reader(fi):
        >>>         yield line
        """

    def preprocess_line(self, line: list) -> list:
        r"""
        Process a line. The structure of each line is exactly
        the same returned by the __iter__ method. Here you should do data preparation
        for the actual model being trained. This is a good place to do batch tokenization,
        padding and so on. If you want to prepare data somewhere else, just return `line`.

        >>> sentences = line[0]
        >>> results = self.hyperparameters.tokenizer.encode_plus(line[0], line[1],
                                                         truncation=True,
                                                         add_special_tokens=True,
                                                         padding='max_length',
                                                         max_length=128)
        >>> return results
        """
        return line

    @staticmethod
    def add_adapter_specific_args(parser: ArgumentParser) -> ArgumentParser:
        r""" Add here arguments that will be available from the command line. """
