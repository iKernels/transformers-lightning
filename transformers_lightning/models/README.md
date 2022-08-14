# Models

This package containts two high-level models that can be used to inherit some useful methods and a general standardized structure.


## TransfomersModel

`TransformersModel` only overrides `configure_optimizers` by returning a better optimizer and the relative scheduler and finally provides a `add_argparse_args` to automatically add the parameters of the optimizer to the global parser.

Example:
```python
>>> parser = ArgumentParser()
>>> TransformerModel.add_argparse_args(parser)
>>> save_transformers_callback = callbacks.TransformersModelCheckpointCallback(hyperparameters)
>>> hyperparameters = parser.parse_args()
```
