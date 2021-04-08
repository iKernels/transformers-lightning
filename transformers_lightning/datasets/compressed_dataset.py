from compressed_dictionary import CompressedDictionary

from transformers_lightning.datasets.super_dataset import SuperDataset


class CompressedDataset(SuperDataset):
    r"""
    Superclass of all dataset using new `CompressedDictionary`s.
    Dataset is completely read into memory in a compressed way and decompression is done on-the-fly.
    """

    def __init__(self, hparams, filepath: str):
        super().__init__(hparams)
        self.data = CompressedDictionary.load(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        """ Get dict of data at a given position. """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        assert 0 <= idx < len(self), (f"Received index out of range {idx}, range: {0} <= idx < {len(self)}")

        return self.data[idx]