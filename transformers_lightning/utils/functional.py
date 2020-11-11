import math
import os

import torch


def remove_from_dict(dictionary, keys=[]):
    for k in keys:
        assert k in dictionary, f"{k} not in dict {dictionary}"
        del dictionary[k]

def all_equal_in_iterable(iterable):
    if len(iterable) == 0:
        return True
    else:
        return iterable.count(iterable[0]) == len(iterable)

def all_equal_length_in_iterable(iterable):
    if len(iterable) == 0:
        return True
    else:
        target_length = len(iterable[0])
        return all([len(x) == target_length for x in iterable])

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def get_inner_type(parameter_list):
    parameter_type = None
    for param in parameter_list:
        if parameter_type is None:
            parameter_type = type(param)
        else:
            assert parameter_type is type(param)
    return parameter_type

def model_checksum(model):
    return sum([x.data.sum() for x in model.parameters()])

def model_gradient_checksum(model):
    return sum([x.grad.sum() if x.grad is not None else 0.0 for x in model.parameters()])

def concat_dict_values(data):
    res = {}
    for dictionary in data:
        for key in dictionary.keys():
            if key in res:
                res[key].append(dictionary[key])
            else:
                res[key] = [dictionary[key]]
    return res

def get_value_from_list_of_dicts(lista, keys):
    # if working with single key?
    single = isinstance(keys, str)
    # give always a list to the last part
    if single:
        keys = [keys]
    # cannot concatenate empty list
    assert len(lista) > 0, "Cannot concatenate empty list"
    # concat or stack based on input shape
    res = []
    for key in keys:
        to_be_concat = [o[key] for o in lista]
        if len(to_be_concat[0].shape) < 1:
            res.append(torch.stack(to_be_concat))
        else:
            res.append(torch.cat(to_be_concat))
    return res[0] if single else res

def split(a, n) -> tuple:
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_base_model(model):
    while(hasattr(model, "module")):
        model = model.module
    return model

def safe_merge(dict_1, dict_2) -> dict:
    """
    Merge two dictionaries asserting that they do not have common keys
    """
    res = dict()
    for key, value in dict_1.items():
        res[key] = value

    for key, value in dict_2.items():
        assert not key in res, f"Key {key} is duplicate"
        res[key] = value
    return res

def check_batch_sizes(batch_size, mini_batch_size) -> None:
    # Training batch size
    assert isinstance(batch_size, int), "batch_size must be an integer"
    assert isinstance(mini_batch_size, int), "mini_batch_size must be an integer"

    # Assert batch size is divisible by mini batch size
    assert batch_size % mini_batch_size == 0, \
        f"batch_size {batch_size} must be divisible by mini_batch_size {mini_batch_size}"

def collate_single_fn(data):
    """ Merge n dicts with identical keys creating list of values. """
    res = concat_dict_values(data)
    # convert values to tensors
    res = {k: torch.tensor(v) for k, v in res.items()}
    return res

def collate_multi_fn(data):
    """ Concatenate the values of each batch dict of each dataset. """
    res = [collate_single_fn(x) for x in zip(*data)]
    return res

def add_base_path_to_metrics(dictionary, base_path):
    # clean base path
    if base_path.startswith("/"):
        base_path = base_path[1:]
    if base_path.endswith("/"):
        base_path = base_path[:-1]
    # add base_path to every string
    return {f"{base_path}/{k}": v for k, v in dictionary.items()}

def concat_generators(*args):
    for gen in args:
        yield from gen

def _split_batch(data: torch.Tensor, optimizer_idx: int, iter_per_step: int) -> torch.Tensor:
    if data is None:
        return None
    # find part size
    part_len = (data.shape[0] // iter_per_step)
    if optimizer_idx < (iter_per_step - 1):
        return data[part_len * optimizer_idx : part_len * (optimizer_idx + 1)]
    else:
        return data[part_len * optimizer_idx :]

def split_batch(*args, optimizer_idx: int = 0, iter_per_step: int = 1) -> tuple:
    return tuple(_split_batch(x, optimizer_idx, iter_per_step) for x in args)

def use_token_type_ids(model_name):
    return all([x not in model_name for x in ["xlm", "roberta", "camembert", "distilbert"]])

def join_on_path(files_list, base_path):
    """
    Join a list of files on a given path
    """
    assert isinstance(files_list, list) and isinstance(base_path, str), \
        "files_list must be a list (found {}) and base_path must be a string (found {})".format(
            type(files_list), type(base_path))
    return [os.path.join(base_path, f) for f in files_list]

def uniform_distrib(size, **kwargs):
    """ Return a vector containing uniform values such that the last exis sums to 1.0. """
    return torch.full(size=size, fill_value=1/size[-1], **kwargs)

def normalization(x, dim=-1):
    """ Do (x - E[x]) / ( E[x - E[x]] ) on axis specified by dim. """
    return torch.true_divide(
        torch.sub(x, x.mean(dim=dim).unsqueeze(1)), 
        x.std(dim=dim).unsqueeze(1)
    )

def concat_dict_tensors(*args, dim=0):
    """ Concat dictionaries containing tensors on `dim` dimension. """
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

def filter_generator(generator_in, step, offset):
    """
    Return elements from a generator. First `offset` elements are discarded
    Then, return an element after every every `step` extracted
    """

    assert step is not None and step >= 0, f"step must be non-negative, found {step}"
    assert offset is not None and offset >= 0, f"offset must be non-negative, found {offset}"

    # advance to the target offset and return first element
    for _ in range(offset):
        try:
            next(generator_in)
        except:
            return
    try:
        yield next(generator_in)
    except:
        return

    while True:
        # consume world_size - 1 inputs
        for _ in range(step - 1):
            try:
                next(generator_in)
            except:
                return
        try:
            yield next(generator_in)
        except:
            return
