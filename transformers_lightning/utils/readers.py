import json
import os
from argparse import Namespace
from typing import Dict, Generator, Iterable, Union

import yaml
from pytorch_lightning import _logger as logger


def load_yaml(filename: str, to_namespace: bool = True) -> Union[Dict, Namespace]:
    r"""
    Load a yaml file from disk
    """
    assert os.path.isfile(filename), f"File {filename} does not exist!"
    with open(filename, 'r') as infile:
        res = yaml.safe_load(infile.read())
    if to_namespace:
        return Namespace(**res)
    return res


def load_json(filename: str, to_namespace: bool = True) -> Union[Dict, Namespace]:
    r"""
    Load config from json file from disk
    """
    assert os.path.isfile(filename), f"File {filename} does not exist!"
    with open(filename, 'r') as infile:
        res = json.load(infile)
    if to_namespace:
        return Namespace(**res)
    return res


def dump_json(filename: str, data: Dict, complain: bool = False) -> None:
    r"""
    Save json to file.
    """
    if os.path.isfile(filename):
        if complain:
            raise ValueError(f"File {filename} does already exist!")
        else:
            logger.warn(f"Overwriting {filename} file!")

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def strip_lines(iterable: Iterable[str]) -> Generator:
    r"""
    Remove blank lines from iterable over strings and return new generator.
    """
    for line in iterable:
        if line.strip():
            yield line
