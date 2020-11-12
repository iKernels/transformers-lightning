import torch

from torch.utils.data import IterableDataset
from transformers_lightning import utils
from transformers_lightning.datasets import SuperTransformersDataset


class TransformersIterableDataset(SuperTransformersDataset, IterableDataset):
    """
    Superclass of all iterable datasets. Tokenization is performed on the fly.
    Dataset is read on the fly from disk to save memory.

    Example
    no distributed training (idx)
    pid default: 0, 1, 2, 3, 4, 5, ..., get_length-1

    with num_workers > 0: example with num_workers=4
    pid 0: 0, 4, 8, 12, ...
    pid 1: 1, 5, 9, 13, ...
    pid 2: 2, 6, 10, 14, ...
    pid 3: 3, 7, 11, 15, ...

    in distributed training with world_size=2 and num_workers=4
    proc 0: 0, 2, 4, 6, 8, 10, ...
    proc 0, worker_pid 0: 0, 8, 16, 24, ...
    proc 0, worker_pid 1: 2, 10, 18, 26, ...
    proc 0, worker_pid 2: 4, 12, 20, 28, ...
    proc 0, worker_pid 3: 6, 14, 22, 30, ...

    proc 1: 1, 3, 5, 7, 9, 11, ...
    proc 1, worker_pid 0: 1, 9, 17, 25, ...
    proc 1, worker_pid 1: 3, 11, 19, 27, ...
    proc 1, worker_pid 2: 5, 13, 21, 29, ...
    proc 1, worker_pid 3: 7, 15, 23, 31, ...
    """

    @property
    def length(self):
        """
        Even if this is an IterableDataset, length may be computed by scrolling the document
        without pre-processing in a fast way. However, this cannot set in __len__ method because
        in that way it may be misleaded for a normal index-based MapDataset
        """
        if not hasattr(self, "_length"):
            self._length = self.get_length()
        return self._length

    def get_length(self):
        """ Get length by doing a fast scan of the input file. """
        reader = SuperTransformersDataset.read_csv_file(self.specs)
        return sum(1 for _ in reader)

    def counter_generator(self, generator_in):
        """ Counter over total number of elements extracted by the generator. """
        self.global_counter = 0
        for x in generator_in:
            yield (x, self.global_counter)
            self.global_counter += 1

    def __iter__(self):
        """
        Return the iterable by nesting different generator, each of which does a different
        filtering based on the process id when in distributed training and on the worker id
        if using also parallel loading in the dataloader.

        - First middlelayer: counter_generator simply increments self.global_counter at every extraction of data
        - Second middlelayer: utils.batch_filter simply ensures that at least `world_size` elements are read at a time
        - Third middlelater: utils.filter_generator on distributed training to filter one element every `world_size`
        - Fourth middlelater: utils.filter_generator on parallel workers processing to filter one element every `num_workers`
        """
        self.reader = SuperTransformersDataset.read_csv_file(self.specs)

        # add counter middlelayer
        self.reader = self.counter_generator(self.reader)

        # add distributed training middlelayer
        if torch.distributed.is_initialized():

            # each node must receive exactly the same data! we must skip something in the end if needed
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            # drop some last elements if last batch would be of different size on some nodes
            self.reader = utils.batch_filter(
                self.reader,
                size=world_size
            )

            # filter away elements destinated to other nodes
            self.reader = utils.filter_generator(
                self.reader,
                step=world_size,
                offset=rank
            )

        # add parallel processing middlelayer
        if torch.utils.data.get_worker_info() is not None:
            self.reader = utils.filter_generator(
                self.reader,
                torch.utils.data.get_worker_info().num_workers,
                torch.utils.data.get_worker_info().id
            )

        for data, idx in self.reader:
            row_dict = self.get_data_as_dict(data)
            row_dict = self.prepare(row_dict, idx=idx)
            yield row_dict

        
