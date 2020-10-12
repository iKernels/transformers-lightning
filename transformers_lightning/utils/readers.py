import json
import os
from argparse import Namespace

import yaml


def load_yaml(filename, to_namespace=True):
    """ Load a yaml file. """
    assert os.path.isfile(filename), f"File {filename} does not exist!"
    with open(filename, 'r') as infile:
        res = yaml.safe_load(infile.read())
    if to_namespace:
        return Namespace(**res)
    return res

def load_json(filename, to_namespace=True):
    """ Load config from json file. """
    assert os.path.isfile(filename), f"File {filename} does not exist!"
    with open(filename, 'r') as infile:
        res = json.load(infile)
    if to_namespace:
        return Namespace(**res)
    return res

def dump_json(filename, data):
    """ Save json to file. """
    assert not os.path.isfile(filename), f"File {filename} does already exist!"
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def strip_lines(iterable):
    """ Remove blank lines from generator. """
    for i, line in enumerate(iterable):
        if line.strip():
            yield line