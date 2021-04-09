import torch
from pytorch_lightning.utilities.distributed import rank_zero_warn

from transformers_lightning import utils


class StackDataset(torch.utils.data.Dataset):
    r"""Dataset as a stack of multiple datasets. (parallel)
    This class is useful to assemble different existing datasets in parallel.
    Arguments:
        datasets (sequence): List of datasets to be stacked
    """

    def __init__(self, *datasets):
        super().__init__()

        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)

        if not utils.functional.all_equal_in_iterable([len(d) for d in self.datasets]):
            rank_zero_warn(
                "Datasets do not have all the same length: "
                ", ".join([f"{d.__class__.__name__}: {len(d)}" for d in self.datasets])
            )

    def __len__(self):
        return min([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(f"absolute value of index {idx} should not exceed dataset length {len(self)}")
            idx = len(self) + idx
        # assert index does not exit boundaries
        assert 0 <= idx < len(self), f"idx with value {idx} exists bounds [{0}, {len(self)}]"
        return tuple(dataset[idx] for dataset in self.datasets)

    def __str__(self):
        res = f"<StackDataset composed of {len(self.datasets)} datasets with lengths "
        res += ", ".join([f"{d.__class__.__name__}: {len(d)}" for d in self.datasets])
        res += ">"
        return res
