# Datasets

`Datasets` are a convinient way to manage data and return the right row only when needed.
This folder contains two main datasets: (i) a simple `MapDataset` that behaves exactly like a list and can be indexed and (ii) a `CompressedDataset` that store data in a `CompressedDictionary` and uses much less memory. It also can be indexes and the length is well-defined. It substitutes the old `IterableDataset` that cause many troubles. The library provides also a simple `StackDataset` to emulate the `zip` command of `python`.

To preprocess each entry of the dataset, the `MapDataset` will call `adapter.preprocess_line` by default. `CompressedDataset`  is instead usually used with already-preprocessed data that are stored in a `CompressedDictionary`.


## MapDataset

A `MapDataset` by default expects only an `Adapter` as input. I will completely read the `adapter` into memory and then it will provide primitives to read the dataset length and for indexing. When doing distributed training, the `Sampler` added by `PyTorch Lightning` will index the right data on each node, making life of the user extremely simple.

To create a MapDataset, you have only to to the following:
```python
trainer = pl.Trainer(...)
adapter = ExampleAdapter(hparams)
dataset = MapDataset(hparams, adapter, trainer=trainer)
```

Normally, you should not implement `Datasets` directly since they are automagically added by [`SuperDataModule`](/transformers-lightning/datamodules).


## CompressedDataset

A `CompressedDataset` reads a `CompressedDictionary` from the disk and then behaves exactly like a `MapDataset` (indexing, length, ...).
See [here](https://github.com/lucadiliello/compressed-dictionary) for more details.


## StackDataset

A simple dataset that emulates the behaviour of the `zip` built-in command in python: it returns a tuple of entries, one for each `MapDataset` that is given as input.

Example:
```python
adapter_1 = ExampleAdapter(hparams)
dataset_1 = MapDataset(hparams, adapter_1)

adapter_2 = ExampleAdapter(hparams)
dataset_2 = MapDataset(hparams, adapter_2)

dataset = StackDataset(dataset_1, dataset_2)

for data_1, data_2 in dataset:
    # data_1 will come from dataset_1
    # data_2 will come from dataset_2
```

`StackDataset` does not inherit from `SuperDataset`!