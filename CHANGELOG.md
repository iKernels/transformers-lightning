# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## 0.6.0

- Metrics has been moved to the `torchmetrics` library ([#81](https://github.com/iKernels/transformers-lightning/issues/81))

- Removed losses package because it has been empty for months.


## 0.5.4

- Language models do not modify `inputs` anymore ([#74](https://github.com/iKernels/transformers-lightning/pull/75))

- All `Language Models` have now a generic `probability` parameter (signature of all language models has been updated).

- Improved efficiency of `ElectraAdamW`.


