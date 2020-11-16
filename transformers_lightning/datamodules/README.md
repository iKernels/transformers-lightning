# DataModules

A `DataModule` is a collection of at most `3` dataset to provide data for both training, validation and testing in a flexible way. For more information about a `DataLoader`, see the original implementation in the [`pytorch-lightning`](https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html) repository. This folder extends the original `DataLoader` with some simple utilities to:
- Do training only then a `train_dataloader` method is defined.
- Same for test.

Moreover, the `SuperDataModule` in this folder is written to accept `Adapters` in input and to use two different type of `Dataset` to offer the maximum flexibility.
The general schema is the following:
- `Adapters` load data from disk / remote storages / invent them / ... and return an iterator along with some funtion to do pre-processing.
- `Datasets` read the data from the `Adapter` and manage indexing, distributed parallel access and so on.
- `DataModules` collect some `Datasets` together and provide them to the training algorithm to do training, validation and test. Moreover, `DataModule` add `DataLoaders` to the datasets and offers some easy primitives to check if it is the case to do training and testing.

A simple `PersonalDataModule` can be defined in either way:

```python
train_adapter = SomeAdapter(hparams, ...)
test_adapter = SomeAdapter(hparams, ...)

PersonalDataModule = SuperDataModule 

datamodule = PersonalDataModule(
    train_adapter=train_adapter,
    test_adapter=test_adapter
)
```

or by defining `Adapters` internally

```python
class PersonalDataModule(SuperDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_adapter = SomeAdapter(self.hparams, ...)
        self.test_adapter = SomeAdapter(self.hparams, ...)

datamodule = PersonalDataModule()
```

## Utilities

Some useful method comprehend:
```python
datamodule.do_train():
    trainer.fit(model, datamodule=datamodule)

datamodule.do_test():
    trainer.test(model, datamodule=datamodule)
```

## Custom `collate_fn`

If your `Adapters` return some strange data structure that is not a simple `dictionary`, you should define the appropriate collate function to merge entries together:
```pytho

train_adapter = SomeAdapterWithCustomOutput(hparams, ...)
test_adapter = SomeAdapterWithCustomOutput(hparams, ...)

PersonalDataModule = SuperDataModule 

datamodule = PersonalDataModule(
    train_adapter=train_adapter,
    test_adapter=test_adapter,
    collate_fn=my_collate_fn
)
```

For more information about `collate_fn`, see the original definition in [`PyTorch DataLoaders`](https://pytorch.org/docs/stable/data.html).

