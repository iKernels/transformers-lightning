# Adapters

An `Adapter` is a standardized interface to read data from disk. Each `Adapter` can be called in the same way, regarding the structure of the dataset that is being read.

An `Adapter` must override at least two methods of the superclass `SuperAdapter`: `__iter__` and `preprocess_line`. The first method should return a simple iterator over all the data in the dataset. The second, that will likely be called in a multiprocessing environment, should do the necessary preprocessing and prepare the data for the neural network. Please move the heavy work, like tokenization, to the `preprocess_line` method because this will be parallelized over many CPU cores. `__iter__` is usually only a function to read from the disk and yield data line by line.

The following example illustrates how to build an `Adapter` to read data from a `tsv` file and how to implement tokenization in the `preprocess_line` method.

The `__iter__` method does not has arguments other than `self`. Parameters should be retrieved through `self.hparams`.
`preprocess_line` instead receives a line at a time and is strongly recommended to return a `dict`. `dict` improves readability and values are automagically concatenated by the `SuperDataModule` class.
