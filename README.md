# transformers-lightning

A collection of `models`, `datasets`, `defaults`, `callbacks`, `loggers`, `metrics` and `losses` to connect the [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html) and the [Transformers](https://huggingface.co/transformers/) libraries.


# Table of contents
**[1. Install](#install)**

**[2. Documentation](#doc)**

  * [2.1. Dataset](#dataset)


<a name="install"></a>
## Install
Install the last stable release with
```
pip install git+https://github.com/lucadiliello/transformers-lightning.git
```


<a name="doc"></a>
## Documentation

<a name="dataset"></a>
### Dataset

Dataset are read by an universal parser that only required a `yaml` file containing the format specs.
The config file should be like the following:
```yaml
# for wikipedia example
filepath: wikipedia_eng/large.tsv # path refers to the defaults.dataset_dir folder
headers: [label, sentence]
delimiter: '\t'
quoting: minimal # one of none, minimal, nonnumeric, all
quotechar: '"'
x: [sentence]   # columns that will be parsed together with the tokenizer
y: [label]      # columns that will be used as labels
```

The dataset should be used in the following way
```python

parser = argsparse.ArgumentParser()
...
# add arguments to the argument parser
parser = DynamicLightningDataModule.add_datamodule_specific_args(parser)
...

hparams = parser.parse_args()

model = ...
trainer = ...

# instantiate the LightningDataModule
datamodule = getDynamicLightningDataModule(hparams)(
    hparams, trainer, model, print_data_preview=True
)

# apply some hacks to remove errors and warnings
utils.hacks(hparams, datamodule)

# Train!
if datamodule.do_train():
    # train function, for example:
    trainer.fit(model, datamodule=datamodule)

# Test!
if datamodule.do_test():
    # test function, for example:
    trainer.test(model, datamodule=datamodule)
```

`getDynamicLightningDataModule` adds the following arguments to the argument parser:
- `--batch_size <int>`: training batch size, default `32`
- `--val_and_test_batch_size <int>`: validation and test batch size, default `256`
- `--num_workers <int>`: number of threads to use when loading data in training/dev/test,
default `multiprocessing.cpu_count()` (number of cpus n the machine)
- `--train_ds <str>`: path of training dataset config with respect to directory `conf/datasets`, default `None`
- `--valid_ds <str>`: path of validation dataset config with respect to directory `conf/datasets`, default `None`
- `--test_ds <str>`: path of test dataset config with respect to directory `conf/datasets`, default `None`
- `--shuffle_train_ds <bool>`: whether the training dataset should be shuffled, default `<true>`
- `--shuffle_valid_ds <bool>`: whether the validation dataset should be shuffled, default `<false>`
- `--shuffle_test_ds <bool>`: whether the test dataset should be shuffled, default `<false>`
- `--chunksize <int>`: whether input file should be read in more chunks of size `<int>` to save memory. We suggest to use this parameter only for dataset larger than `1GB` or `1.000.000` lines. Shuffle will operate only on chunks, no shuffle between different chunks, default `10.000.000`

Notice: all dataset files should be header-free, i.e. header names are specified in the `yaml` file.