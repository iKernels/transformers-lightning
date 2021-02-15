from pytorch_lightning import _logger as logger
import torch
from types import GeneratorType

from torch.utils.data import IterableDataset
from transformers_lightning import utils
from transformers_lightning.datasets import SuperTransformersDataset


class TransformersIterableDataset(SuperTransformersDataset, IterableDataset):
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

    @property
    def length(self):
        r"""
        Even if this is an IterableDataset, length may be computed by scrolling the document
        without pre-processing in a fast way. However, this cannot be set as __len__ attribute because
        in that way it may be misleaded for a normal index-based MapDataset.
        """
        if not hasattr(self, "_length"):
            self._length = sum(1 for _ in iter(self.adapter))
        return self._length

    def __init__(self, *args, start_from_step=None):
        super().__init__(*args)
        r"""
        If `start_from_step` is provided, this dataset will return data
        relative to the `start_from_step`+1 effective step. This is not a problem
        since the training algorithm does not know in advance the total dataset length.
        This applied only to the first epoch, from the following all the data are provided.
        Do not provide a `start_from_step` higher than the number of elements in this
        dataset or higher than the total number of max_steps
        """
        self.start_from_step = None
        if start_from_step is not None:
            assert isinstance(start_from_step, int), (f"`start_from` must be integer, found {start_from_step}")

            total_devices = utils.get_total_devices(trainer=self.trainer)
            effective_batch_size = self.hparams.batch_size * self.hparams.accumulate_grad_batches * total_devices
            self.start_from_step = effective_batch_size * start_from_step

            logger.warning(
                f"IterableDataset starting from step {start_from_step}. If this is the correct "
                f"behaviour, please ignore this warning."
            )

    def __iter__(self):
        r"""
        Return the iterable by nesting different generators, each of which does a different
        filtering based on the process id when in distributed training and on the worker id
        if using also parallel loading in the dataloader.

        0) eventually skip the first `start_from_step` rows to restart from a precise point.
        1) utils.batch_filter simply ensures that at least `world_size` elements are read at a time
        2) utils.filter_generator on distributed training to keep one element every `world_size`
        3) utils.filter_generator on parallel workers processing to keep one element every `num_workers`
        """
        self.reader = iter(self.adapter)

        # skip first steps if needed
        if self.start_from_step is not None:
            self.reader = utils.filter_generator(self.reader, step=1, offset=self.start_from_step)
            self.start_from_step = None

        # add distributed training logic
        if torch.distributed.is_initialized():

            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            self.reader = utils.batch_filter(self.reader, size=world_size)

            self.reader = utils.filter_generator(self.reader, step=world_size, offset=rank)

        # add parallel processing with workers logic
        if torch.utils.data.get_worker_info() is not None:
            self.reader = utils.filter_generator(
                self.reader,
                torch.utils.data.get_worker_info().num_workers,
                torch.utils.data.get_worker_info().id
            )

        # pre-process data and return
        for line in self.reader:

            preprocessed_data = self.adapter.preprocess_line(line)

            # if the adapter return a generator, yield an element per time
            if isinstance(preprocessed_data, GeneratorType):
                yield from preprocessed_data
            else:
                yield preprocessed_data
