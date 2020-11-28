# Models

This package containts two high-level models that can be used to inherit some useful methods and a general standardized structure.


## SuperModel

The `SuperModel` define the basic structure of a `LightningModule` that should be used when training `Transformers` models. It only defines the `configure_optimizers` method by using a simple `AdamW` optimizer and a `max_steps_anyway` that allow to compute the `max_steps` when only `max_epochs` are provided by the user.


## TransfomersModel

`TransformersModel` only overrides `configure_optimizers` by returning a better optimizer and the relative scheduler and finally provides a `add_model_specific_args` to automatically add the parameters of the optimizer to the global parser.

Example:
```python
>>> parser = TransformerModel.add_model_specific_args(parser)
>>> save_transformers_callback = callbacks.TransformersModelCheckpointCallback(hparams)
>>> hparams = parser.parse_args()
```
