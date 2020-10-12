import csv
import os
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import _logger as logger
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer
from transformers_lightning import utils
from transformers_lightning.datasets import SuperTransformersDataset


class TransformersIterableDataset(SuperTransformersDataset, IterableDataset):
    """
    Superclass of all fly datasets. Tokenization is performed on the fly.
    Dataset is split in chunks to save memory using the relative dataframe function.

    Example
    no distributed training (idx)
    pid default: 0, 1, 2, 3, 4, 5, ..., get_length-1

    distributed training (idx) num_workers=4
    pid 0: 0, 4, 8, 12, ...
    pid 1: 1, 5, 9, 13, ...
    pid 2: 2, 6, 10, 14, ...
    pid 3: 3, 7, 11, 15, ...
    """

    @property
    def length(self):
        if not hasattr(self, '_length'):
            self._length = self._get_length()
        return self._length

    def _get_length(self):
        """ Get length by doing a fast scan of the input file. """
        reader = SuperTransformersDataset.read_csv_file(
            self.specs, self.hparams
        )
        res = 0
        for row in reader: res += 1
        return res

    def jump_forward(self, steps: int = 1):
        """ Move reader forward and return last extracted element. """
        row = None
        for i in range(steps):
            row = next(self.reader)
        return row

    """ Init is the same as super class """

    def __iter__(self):
        self.reader = self.__class__.read_csv_file(self.specs, self.hparams)
        self.is_first = True
        if hasattr(self, 'worker_info'):
            delattr(self, 'worker_info') # it may be necessary to reload info after every epoch...

        self.counter = 0

        if self.is_distributed():
            self.jump_forward(steps=self.get_distributed_id())

        return self

    # worker info
    def get_worker_info(self):
        if not hasattr(self, 'worker_info'):
            self.worker_info = torch.utils.data.get_worker_info()
        return self.worker_info

    def is_distributed(self):
        """ Return process id in [0, num_workers-1]! """
        return self.get_worker_info() is not None

    def get_distributed_id(self):
        return self.get_worker_info().id
    
    def get_num_workers(self):
        return self.get_worker_info().num_workers

    def __next__(self):
        """
        Get next element.
        Behaves differently based on whether distributed training is used.
        """
        if self.is_distributed():
            # first step in distributed
            if self.is_first:
                self.is_first = False
                row = self.jump_forward(steps=1)
            # normal step in distributed
            else:
                row = self.jump_forward(steps=self.get_num_workers())

        # normal step in single worker mode
        else:
            row = self.jump_forward(steps=1)

        row_dict = self.get_data_as_dict(row)
        row_dict = self.prepare(row_dict, idx=self.counter)

        self.counter += 1

        return row_dict
