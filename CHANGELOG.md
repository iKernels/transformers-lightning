# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## 0.7.0

- Fixed `metrics` package imports and added tests.


## 0.7.0

- Added `LineAdapter` to read files line by line.

- Every `add_*_specific_args` method now should return nothing.

- Added `predict` capability to `AdaptersDataModule`.

- Added `predict` capability to `CompressedDataModule`.

- Added `do_predict()` and `predict_dataloader()` to `SuperDataModule`.

- Added `do_preprocessing` init argument to `MapDataset` and `IterableDataset` to eventually avoid calling the preprocessing function defined in the `Adapter`.

- Added check over tokenizer type in `whole_word_tails_mask()`.

- Added functions `get_optimizer`, `get_scheduler`, `num_training_steps` and corresponding CLI parameters to `TransformersModel` to allow for more flexible definition of optimizers and schedulers.

- Added optimizer wrappers to be instantiated through CLI parameters. You can still use your own optimizer in `configure_optimizers` without problems.

- Added scheduler wrappers to be instantiated through CLI parameters. You can still use your own scheduler in `configure_optimizers` without problems.

- (Re)Added metrics package with `HitRate`. However, this will likely be moved to `torchmetrics` in the next releases.

- Changed `hparams` attribute of every class (`models`, `adapters`, `datamodules`, `optimizers`, `schedulers`, `callbacks` and `datasets`) to `hyperparameters` to avoid conflict with new lightning `hparams` getters and setters.

- Changed logic of `TransformersModelCheckpointCallback` since training loop has changed in `pytorch-lightning` **v1.4**.

- Removed `TransformersAdapter` because it was too specific and useless.

- General refactoring of classes. Cleaned and removed unused imports. Refactored also some tests.


## 0.6.0

- Added `CompressedDataModule` based on `CompressedDataset`

- Added `CompressedDataset` based on [`CompressedDictionary`](https://github.com/lucadiliello/compressed-dictionary)

- Removed `IterableDataset`

- Metrics has been moved to the `torchmetrics` library ([#81](https://github.com/iKernels/transformers-lightning/issues/81))

- Removed losses package because it has been empty for months.


## 0.5.4

- Language models do not modify `inputs` anymore ([#74](https://github.com/iKernels/transformers-lightning/pull/75))

- All `Language Models` have now a generic `probability` parameter (signature of all language models has been updated).

- Improved efficiency of `ElectraAdamW`.


