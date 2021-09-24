import torch
from pytorch_lightning.utilities.distributed import distributed_available
from torch.utils.data import IterableDataset

from transformers_lightning import utils
from transformers_lightning.datasets.super_dataset import SuperDataset


class TransformersIterableDataset(SuperDataset, IterableDataset):
    r"""
    Superclass of all iterable datasets. Pre-processing is performed on the fly.
    Dataset is read on the fly from disk to save memory.

    When doing distributed training, Lightning does not add a specific sampler
    for IterableDatasets: this means that we should implement the logic to
    let each node have the right portion of data.

    Example
    no distributed training (idx)
    >>> pid default: 0, 1, 2, 3, 4, 5, ..., get_length-1

    with num_workers > 0: example with num_workers=4
    >>> pid 0: 0, 4, 8, 12, ...
    >>> pid 1: 1, 5, 9, 13, ...
    >>> pid 2: 2, 6, 10, 14, ...
    >>> pid 3: 3, 7, 11, 15, ...

    in distributed training with world_size=2 and num_workers=4
    >>> proc 0: 0, 2, 4, 6, 8, 10, ...
    >>> proc 0, worker_pid 0: 0, 8, 16, 24, ...
    >>> proc 0, worker_pid 1: 2, 10, 18, 26, ...
    >>> proc 0, worker_pid 2: 4, 12, 20, 28, ...
    >>> proc 0, worker_pid 3: 6, 14, 22, 30, ...

    >>> proc 1: 1, 3, 5, 7, 9, 11, ...
    >>> proc 1, worker_pid 0: 1, 9, 17, 25, ...
    >>> proc 1, worker_pid 1: 3, 11, 19, 27, ...
    >>> proc 1, worker_pid 2: 5, 13, 21, 29, ...
    >>> proc 1, worker_pid 3: 7, 15, 23, 31, ...

    Moreover, one must ensure that each node receives exactly the same number of data.
    This is not allowed and may lead to a crash in the distributed training:
    >>> proc 0: 0, 2, 4, 6, 8
    >>> proc 1: 1, 3, 5, 7, 9, 11
    This can be solved by reading at least `world_size` (2 in this case) elements for
    each iteration from the adapter.
    """

    def __iter__(self):
        r"""
        Return the iterable by nesting different generators, each of which does a different
        filtering based on the process id when in distributed training and on the worker id
        if using also parallel loading in the dataloader.

        1) utils.batch_filter simply ensures that at least `world_size` elements are read at a time
        2) utils.filter_generator on distributed training to keep one element every `world_size`
        3) utils.filter_generator on parallel workers processing to keep one element every `num_workers`
        """
        reader = iter(self.adapter)

        # add distributed training logic
        if distributed_available():

            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            reader = utils.batch_filter(reader, size=world_size)
            reader = utils.filter_generator(reader, step=world_size, offset=rank)

        # add parallel processing with workers logic
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            reader = utils.filter_generator(reader, step=worker_info.num_workers, offset=worker_info.id)

        # pre-process data and return
        for line in reader:
            if self.do_preprocessing:
                line = self.adapter.preprocess_line(line)
            yield line
