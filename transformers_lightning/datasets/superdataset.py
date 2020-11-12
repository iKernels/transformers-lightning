import copy
import csv
import os
from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import _logger as logger
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer
from transformers_lightning import utils
from transformers_lightning.datasets import QUOTING_MAP


class SuperTransformersDataset:
    """
    Superclass of all fly datasets. Tokenization is performed on the fly.
    Dataset is split in chunks to save memory using the relative dataframe function.
    """

    @staticmethod
    def process_line(line, specs):
        """ Convert fields in list to int, float or bool if possible. """
        res = []
        try:
            for name, entry in zip(specs.names, line):
                if name not in specs.x:
                    try:
                        res.append(eval(entry))
                    except:
                        res.append(entry)
                else:
                    res.append(entry)
            return res
        except:
            with open("out.log", "w") as fo:
                fo.write(f"Debugging dataset: {specs.names}, {line}")
            exit(1)

    @staticmethod
    def check_and_prepare_dataset_specs(specs, hparams):
        """ Check that YAML file has been built correctly. """
        specs = copy.deepcopy(specs)

        # x and y should be lists
        if not isinstance(specs.x, list):
            specs.x = [specs.x]
        if not isinstance(specs.y, list):
            specs.y = [specs.y]

        if hasattr(specs, "quoting"):
            assert specs.quoting in QUOTING_MAP, (
                f"specs.quoting not valid among {QUOTING_MAP}, got {specs.quoting} instead"
            )
            specs.quoting = QUOTING_MAP[specs.quoting]

        # check file exists
        specs.filepath = os.path.join(hparams.dataset_dir, specs.filepath)
        assert os.path.isfile(specs.filepath), f"Dataset file {specs.filepath} does not exist!"

        specs.names = [n.lower() for n in specs.names]

        # fix esscaping
        if hasattr(specs, "delimiter"):
            specs.delimiter = str.encode(specs.delimiter).decode('unicode_escape')

        # assertions
        assert len(specs.x) <= 2, "x length higher than 2 not supported yet"
        assert len(specs.x) == len(set(specs.x)), "names must be unique"
        for _x in specs.x:
            assert _x in specs.names, \
                f"key {_x} of x is not part of names {specs.names}"
        
        assert len(specs.y) == len(set(specs.y)), "names must be unique"
        for _y in specs.y:
            assert _y in specs.names, \
                f"key {_y} of y is not part of names {specs.names}"

        return specs

    @staticmethod
    def read_csv_file(specs, hparams):
        """ Return a generator of parsed lines. """
        kwargs = {}
        for param in ['delimiter', 'quoting', 'quotechar']:
            if hasattr(specs, param):
                kwargs[param] = getattr(specs, param)

        with open(specs.filepath, "r") as fi:
            # use utils.strip_lines to emulate skip_blank_lines of pd.DataFrame
            reader = csv.reader(utils.strip_lines(fi), **kwargs)
            for line in reader:
                yield SuperTransformersDataset.process_line(line, specs)

    def __init__(self,
        hparams: Namespace,
        tokenizer: PreTrainedTokenizer,
        specs,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.hparams = hparams
        self.tokenizer = tokenizer
        # clean specs
        self.specs = self.__class__.check_and_prepare_dataset_specs(specs, hparams)

    def get_data_as_dict(self, row):
        return { k: row[i] for i, k in enumerate(self.specs.names)}

    def prepare(self, row_dict, idx=None) -> dict:
        """
        Prepare data to be returned. Should return a tuple of lists.
        """
        # get columns of interest and tokenizer them in pairs
        sentences = [row_dict[x] for x in self.specs.x]
        results = self.tokenizer.encode_plus(*sentences,
                                            truncation=True,
                                            add_special_tokens=True,
                                            padding='max_length',
                                            max_length=self.hparams.max_sequence_length)
        # add ids field
        if 'ids' not in row_dict and idx is not None:
            results["ids"] = idx
        else:
            results["ids"] = row_dict["ids"]

        # add label fields
        for label in self.specs.y:
            results[label] = np.array([row_dict[label]], dtype=np.int64)

        return results
