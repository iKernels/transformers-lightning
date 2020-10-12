# transformers-lightning

A collection of `models`, `datasets`, `defaults`, `callbacks`, `loggers`, `metrics` and `losses` to connect the [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html) and the [Transformers](https://huggingface.co/transformers/) libraries.


# Table of contents
**[1. Install](#install)**

**[2. Documentation](#doc)**
  
  * [2.1. Datasets](#datasets)
  * [2.2. Datamodules](#datamodules)


<a name="install"></a>
## Install
Install the last stable release with
```
pip install git+https://github.com/lucadiliello/transformers-lightning.git --upgrade
```


<a name="doc"></a>
## Documentation

<a name="dataset"></a>
### Dataset

Dataset are read by an universal `*sv` (tsv, csv, ...) parser that only required a `yaml` file containing the format specifications.
The config file should be like the following:
```yaml
# for wikipedia example
filepath: wikipedia_eng.tsv # path refers to the defaults.dataset_dir folder
names: [label, sentence]
delimiter: '\t'
quoting: minimal # one of none, minimal, nonnumeric, all
quotechar: '"'
x: [sentence]   # columns that will be parsed together with the tokenizer
y: [label]      # columns that will be used as labels
```
Put all `yaml` files with respect to your datasets in the `defaults.config_dir/datasets` folder.
Notice: all dataset files should be header-free, i.e. header names are specified in the `yaml` file with the `names` parameter.


<a name="datamodules"></a>
### Datamodules

`LightningDataModules` are an easy way to collect some dataset together and to allow reusability.
By default this library loads a `train`, `val` and `test` dataset. All three datasets are optional.

Example:
```python
from transformers_lightning.datamodules import SuperDataModule

class QuoraDataModule(SuperDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_config = "quora_train.yaml"
        self.val = "quora_valid.yaml"
        self.test_config = "quora_test.yaml"

    train_dataloader = SuperDataModule.default_train_dataloader
    val_dataloader = SuperDataModule.default_val_dataloader
    test_dataloader = SuperDataModule.default_test_dataloader
```

By default, all dataset are loaded with a `TransformersMapDataset`, this means that each dataset is
completely loaded in memory and shuffled. If you don't want to load all dataset completely in memory,
use the argument `--dataset_style 'iter'`.
