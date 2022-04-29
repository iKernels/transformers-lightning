# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## 0.7.9

- Fixed steps computation when `max_steps` is not provided by the user.

- Added `JsonboardLogger`.

- Added some tests for automatic steps computation with `deepspeed`.


## 0.7.8

- Fixed `TransformersMapDataset` parameters and adapter loading.

- Removed `CompressedDataModule`.

- Added `RichProgressBar` with `global_step` logging.

- Fixed deprecated `transformers` `AdamW` inside optimizers to `torch` implementation.

- Fixed typos.


## 0.7.7

- Removed update `TransformersModelCheckpointCallback`.

- `TransformersModel.num_training_steps` is not a function and not a property anymore + fix.

- Updated tests to use new `accelerator` and `strategy` signature for defining the training hardware to be used.

- Fixed check on shuffle in `SuperDataModule`.

- Completely removed metrics package, now all metrics available in `torchmetrics` library.


## 0.7.6

- Package publication fixed


## 0.7.5

- Added `trainer` as second positional argument of every DataModule.

- Renamed `MapDataset` to `TransformersMapDataset`.

- Fixed typo about default shuffling in `SuperDataModule` and `CompressedDataModule`.


## 0.7.4

- Added `SortingLanguageModeling` technique and tests.

- Added `SwappingLanguageModeling` technique and tests.

- Added `add_adapter_specific_args` method to `SuperAdapter` to allow adding parameters to the CLI.

- Fixed typo with which `AdapterDataModule` was not receiving `collate_fn` argument.

- Fixed typos in `imports`.

- Refactored `datamodules` section.


## 0.7.3

- Added `get_dataset` method to `AdaptersDataModule` to facilitate creation of dataset from adapters.

- Dropped support for `drop_last` in every dataloader: lightning uses `False` everywhere by default.

- Fixed `TransformersModel.num_training_steps` that in some cases was providing slightly wrong numbers due to rounding.

- Fixed `whole_words_tail_mask` in `language_modeling` which was not working correctly.

- Improved testing of `models` and `language_models`.


## 0.7.2

- Added tests for `optimizers` package.

- Fixed some imports.

- Fixed some calls to **super** method in optimizers and schedulers.


## 0.7.1

- Fixed `metrics` package imports and added tests.


## 0.7.0

- Added `LineAdapter` to read files line by line.

- Every `add_*_specific_args` method now should return nothing.

- Added `predict` capability to `AdaptersDataModule`.

- Added `predict` capability to `CompressedDataModule`.

- Added `do_predict()` and `predict_dataloader()` to `SuperDataModule`.

- Added `do_preprocessing` init argument to `MapDataset` and `TransformersIterableDataset` to eventually avoid calling the preprocessing function defined in the `Adapter`.

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


