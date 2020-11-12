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
        return self._length

    def parse_and_return_length(self):
        """ Get length by doing a fast scan of the input file. """
        reader = SuperTransformersDataset.read_csv_file(
            self.specs, self.hparams
        )
        return sum(1 for _ in reader)

    def counter_generator(self, generator_in):
        """ Counter over total number of elements extracted by the generator. """
        for x in generator_in:
            yield x
            self.global_counter += 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._length = self.parse_and_return_length()

    def __iter__(self):
        self.reader = SuperTransformersDataset.read_csv_file(
            self.specs, self.hparams
        )
        self.global_counter = 0

        # add counter middlelayer
        self.reader = self.counter_generator(self.reader)
        self.limit = None

        # add distributed training middlelayer
        if torch.distributed.is_initialized():

            # each node must receive exactly the same data! we must skip something in the end if needed
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            if (self._length % world_size) != 0:
                # BUG in lightning -> must require that every node has something to put in next batch
                self.limit = (self._length // world_size) * world_size
                print(f"WARNING: dataset length limited to the greatest multiple of the world size ({world_size}): {self.limit}")

            self.reader = utils.filter_generator(
                self.reader,
                world_size,
                rank
            )

        # add parallel processing middlelayer
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.reader = utils.filter_generator(
                self.reader,
                worker_info.num_workers,
                worker_info.id
            )

        for row in self.reader:
            # if limit is 
            if self.limit is not None and self.global_counter >= self.limit:
                return

            row_dict = self.get_data_as_dict(row)
            row_dict = self.prepare(row_dict, idx=self.global_counter)
            row_dict["ids"] = self.global_counter
            yield row_dict

        
