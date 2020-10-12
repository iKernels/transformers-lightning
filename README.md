# transformers-lightning

A collection of `models`, `datasets`, `defaults`, `callbacks`, `loggers`, `metrics` and `losses` to connect the [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html) and the [Transformers](https://huggingface.co/transformers/) libraries.


# Table of contents
**[1. Install](#install)**

**[2. Documentation](#doc)**
  
  * [2.1. Datasets](#datasets)
  * [2.2. Datamodules](#datamodules)

**[3. Main file](#main)**


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


<a name="main"></a>
## Main file

Structure your main file like:

```python

import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

import models
import datamodules

from transformers_lightning import utils, callbacks, datamodules


# Print high precision tensor values
torch.set_printoptions(precision=16)

def main(hparams):

    # instantiate PL model
    model = pl_model_class(hparams)

    utils.init_folders(hparams)

    # default tensorboard logger
    test_tube_logger = pl.loggers.TestTubeLogger(
        os.path.join(hparams.output_dir, hparams.tensorboard_dir), name=hparams.name)

    # Save pre-trained models to
    save_transformers_callback = callbacks.TransformersModelCheckpointCallback(hparams)

    # instantiate PL trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        default_root_dir=hparams.output_dir,
        profiler=True,
        logger=test_tube_logger,
        callbacks=[save_transformers_callback],
        log_gpu_memory='all',
        weights_summary='full'
    )

    # Datasets
    datamodule = pl_datamodule_class(hparams, model, trainer)

    # Train!
    if datamodule.do_train():
        trainer.fit(model, datamodule=datamodule)

    # Test!
    if datamodule.do_test():
        trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':

    # configurations
    from transformers_lightning.defaults import DefaultConfig

    # list available models and datamodules
    all_models = utils.get_classes_from_module(models, parent=pl.LightningModule)
    all_datamodules = utils.get_classes_from_module(datamodules, parent=pl.LightningDataModule)

    parser = ArgumentParser()

    # Global level parameters (model and data)
    parser.add_argument('-m', '--model', type=str, required=True, choices=all_models.keys())
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=all_datamodules.keys())

    # Experiment name, used both for checkpointing, pre_trained_names, logging and tensorboard
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the model')

    # I/O folders
    parser = DefaultConfig.add_defaults_args(parser)

    # retrieving model with temporary parsered arguments
    tmp_params, extra = parser.parse_known_args()

    # get pl_model_class in advance to know which params it needs, same for the datamodule
    pl_model_class = all_models[tmp_params.model]
    pl_datamodule_class = all_datamodules[tmp_params.dataset]

    # add model specific args
    parser = pl_model_class.add_model_specific_args(parser)
    parser = pl_datamodule_class.add_datamodule_specific_args(parser)

    # add callback / logger specific parameters
    parser = callbacks.TransformersModelCheckpointCallback.add_callback_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    # get NameSpace of paramters
    hparams = parser.parse_args()

    main(hparams)

```
