from pytorch_lightning import Trainer

from transformers_lightning.adapters.super_adapter import SuperAdapter
from transformers_lightning.datasets.super_dataset import SuperDataset


class MapDataset(SuperDataset):
    r"""
    Superclass of all map datasets. Tokenization is performed on the fly.
    Dataset is completely read into memory.
    """

    def __init__(self, hyperparameters, adapter: SuperAdapter, trainer: Trainer):
        super().__init__(hyperparameters, adapter=adapter, trainer=trainer)
        self.data = list(iter(self.adapter))

    def __len__(self):
        return len(self.data)

    def _get_sample(self, idx):
        """ Get dict of data at a given position without preprocessing. """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        assert 0 <= idx < len(self), (f"Received index out of range {idx}, range: {0} <= idx < {len(self)}")

        row = self.data[idx]
        return row

    def __getitem__(self, idx) -> dict:
        """ Get dict of data at a given position. """
        row = self._get_sample(idx)
        if self.do_preprocessing:
            row = self.adapter.preprocess_line(row)
        return row
