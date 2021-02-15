import os
from typing import Dict, Generator, Iterable, List, Tuple, Union

import torch


def remove_from_dict(dictionary: Dict, keys: List[str] = []) -> Dict:
    r"""
    Safely remove a list of keys from a dictionary
    """
    for k in keys:
        assert k in dictionary, f"{k} not in dict {dictionary}"
        del dictionary[k]


def all_equal_in_iterable(iterable: Iterable) -> bool:
    r"""
    Check all items in an iterable 
    """
    if len(iterable) == 0:
        return True
    else:
        return iterable.count(iterable[0]) == len(iterable)


def flatten(list_of_lists: List[List]) -> List:
    r"""
    Unroll a list of lists
    """
    return [item for sublist in list_of_lists for item in sublist]


def get_inner_type(parameter_list: List) -> type:
    r"""
    Check both that all entries in a list are of the same and return that type
    """
    parameter_type = None
    for param in parameter_list:
        if parameter_type is None:
            parameter_type = type(param)
        else:
            assert parameter_type is type(param)
    return parameter_type


def model_checksum(model: torch.nn.Module) -> torch.Tensor:
    r"""
    Return a checksum of a torch model by summing all the paramter values.
    This is a useful indicator of whether the model has been updated while training.
    Be sure to print many significant digits to see substantial differences.
    """
    return sum([x.data.sum() for x in model.parameters()])


def model_gradient_checksum(model: torch.nn.Module) -> torch.Tensor:
    r"""
    Return a checksum of the gradients of a torch model by summing all the gradient values.
    This is a useful indicator of whether the model is backpropagating correctly while training.
    Be sure to print many significant digits to see substantial differences.
    """
    return sum([x.grad.sum() if x.grad is not None else 0.0 for x in model.parameters()])


def concat_dict_values(data: List) -> Dict:
    r"""
    Given a list of dictionaries with the same keys, return a dictionary in which
    each value is the contatenation of the values in the original dictionaries.
    """
    res = {}
    for dictionary in data:
        for key in dictionary.keys():
            if key in res:
                res[key].append(dictionary[key])
            else:
                res[key] = [dictionary[key]]
    return res


def split(a: List, n: int) -> tuple:
    r"""
    Split a list in `n` equal parts (or similar, if `len(a) % n != 0`)
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def safe_merge(dict_1: Dict, dict_2: Dict) -> Dict:
    r"""
    Merge two dictionaries asserting that they do not have common keys
    """
    res = dict()
    for key, value in dict_1.items():
        res[key] = value

    for key, value in dict_2.items():
        assert not key in res, f"Key {key} is duplicate"
        res[key] = value
    return res


def collate_single_fn(data: List[Dict]) -> Dict[str, torch.Tensor]:
    r"""
    Merge n dicts with identical keys creating list of value tensors.
    """
    res = concat_dict_values(data)
    # convert values to tensors
    res = {k: torch.tensor(v) for k, v in res.items()}
    return res


def collate_multi_fn(data):
    r"""
    Concatenate the values of each batch dict of each dataset when using, for example, a `StackDataset`
    """
    res = [collate_single_fn(x) for x in zip(*data)]
    return res


def concat_generators(*args: Iterable[Generator]) -> Generator:
    r"""
    Concat generators by yielding from first, second, ..., n-th
    """
    for gen in args:
        yield from gen


def join_on_path(files_list: List[str], base_path: str) -> List[str]:
    r"""
    Join a list of files on a given path
    """
    assert isinstance(files_list, list) and isinstance(base_path, str), \
        "files_list must be a list (found {}) and base_path must be a string (found {})".format(
            type(files_list), type(base_path))
    return [os.path.join(base_path, f) for f in files_list]


def uniform_distrib(size: Union[Tuple, List, torch.Size], **kwargs) -> torch.Tensor:
    r"""
    Return a vector containing uniform values such that the last axis sums to 1.0.
    """
    return torch.full(size=size, fill_value=(1 / size[-1]), **kwargs)


def normalize_standard(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""
    Do (x - E[x]) / ( E[x - E[x]] ) on axis specified by dim.
    """
    return torch.where(
        (x.std(dim=dim) > 0).unsqueeze(-1), (x - x.mean(dim=dim).unsqueeze(-1)) / x.std(dim=dim).unsqueeze(-1),
        (x - x.mean(dim=dim).unsqueeze(-1))
    )


def normalize_linear(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""
    Do (x - min(x)) / (max(x) - min(x)) on axis specified by dim.
    """
    max_x = torch.max(x, dim=dim)
    min_x = torch.min(x, dim=dim)
    return (x - min_x) / (max_x - min_x)


def concat_dict_tensors(*args: Iterable[Dict], dim: int = 0) -> Dict[str, torch.Tensor]:
    r"""
    Concat dictionaries values containing tensors on `dim` dimension.
    """
    if (len(args) == 1) and isinstance(args[0], list):
        args = args[0]

    res = {}
    for dictionary in args:
        for key in dictionary.keys():
            if key in res:
                res[key].append(dictionary[key])
            else:
                res[key] = [dictionary[key]]

    res = {k: torch.cat(v, dim=dim) for k, v in res.items()}
    return res
