# Datasets

`Datasets` are a convinient way to manage dataset and process and return the right data only when needed.
This folder contains two main datasets: (i) a simple `MapDataset` that behaves exactly like a list and can be indexed and (ii) an `IterableDataset` that has no well-defined length and continues to preprocess and provide data until the corresponding adapter is completely emptied.

To preprocess each entry of the dataset, the `Dataset` will call `adapter.preprocess_line` by default.

## MapDataset

A `MapDataset` by default expects only an `Adapter` as input. I will completely read the `adapter` into memory and then it will provide primitives to read the dataset length and for integer indexing. When doing distributed training, the `Sampler` added by `PyTorch Lightning` will index the right data on each node, making life of the user extremely simple.

To create a MapDataset, you have only to to the following:
```python
adapter = ExampleAdapter(hparams)
dataset = MapDataset(hparams, adapter)
```

Normally, you should not implement `Datasets` directly since they are automagically added by [`SuperDataModule`](/transformers-lightning/datamodules).


## IterableDataset

An `IterableDataset` is far more interesting. It contains the logic to read and preprocess a dataset on the fly while doing training by distributing the workload on many processes (by default as many processes as the number of logic threads of your CPU). Since it is not possible to know the dataset length in advance (as with generators and iterators), it includes the logic to provide different samples to all the different nodes of a distributed execution.

In particular, the `IterableDataset` return the data in the following way (index are shown):

Single node training (no 'ddp' neither 'dp'), `num_workers = 0` or `num_workers = 1`, check [`this`](https://pytorch.org/docs/stable/data.html) for more info about workers.
```bash
>>> worker_pid 0 or 1: 0, 1, 2, 3, 4, 5, ..., length-1
```

Single node with many workers to preprocess data (`num_workers = 4`):
```bash
>>> worker_pid 0: 0, 4, 8, 12, ...
>>> worker_pid 1: 1, 5, 9, 13, ...
>>> worker_pid 2: 2, 6, 10, 14, ...
>>> worker_pid 3: 3, 7, 11, 15, ...
```

Distributed training with `world_size = 2` and `num_workers = 4` for each node:
```bash
>>> proc 0: 0, 2, 4, 6, 8, 10, ...
>>> # distributed in the following way across workers:
>>> proc 0, worker_pid 0: 0, 8, 16, 24, ...
>>> proc 0, worker_pid 1: 2, 10, 18, 26, ...
>>> proc 0, worker_pid 2: 4, 12, 20, 28, ...
>>> proc 0, worker_pid 3: 6, 14, 22, 30, ...

>>> proc 1: 1, 3, 5, 7, 9, 11, ...
>>> # distributed in the following way across workers:
>>> proc 1, worker_pid 0: 1, 9, 17, 25, ...
>>> proc 1, worker_pid 1: 3, 11, 19, 27, ...
>>> proc 1, worker_pid 2: 5, 13, 21, 29, ...
>>> proc 1, worker_pid 3: 7, 15, 23, 31, ...
```

Moreover, one must ensure that each node receives exactly the same number of data.
This is not allowed and may lead to a crash in the distributed training:
```
>>> proc 0: 0, 2, 4, 6, 8
>>> proc 1: 1, 3, 5, 7, 9, 11
```
This can be solved by reading at least `world_size` (2 in this case) elements for each iteration from the adapter. So be warned: if you provide an Adapter to the distributed session with a number of elements that is not and exact multiple of the `world_size`, a small quantity of data in the last batches will be dropped!

The `IterableDataset` offers also a last **experimental** feature: jump the first `start_from_step` steps to resume exactly from a checkpoint. This will compute the effective `batch_size` by taking into account the number of nodes in distributed training and the `accumulate_grad_batches` parameter and will skip data read in the first `start_from_step` steps.

```python
adapter = ExampleAdapter(hparams)
dataset = IterableDataset(hparams, adapter, start_from_step=10)
```

Please do not provide a `start_from_step` longer than an epoch!