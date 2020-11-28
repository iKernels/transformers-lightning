# Callbacks

Callbacks are classes that are regularly called in every part of the training, validation and testing phase. Callbacks are used to save checkpoints, log data and o regular checks.


## TransformersModelCheckpointCallback

This class allow for a fine-grained checkpointing of models of the [`transformers`](https://github.com/huggingface/transformers) library when trained with the [`pytorch_lightning`](https://github.com/PyTorchLightning/pytorch-lightning) framework.

This callback can be used to save a checkpoint after every `k` steps, after every epoch and/or after every validation loop. To add the parameters of this callback to the training `hparams`, do:

```python
>>> parser = ArgumentParser()
>>> ...
>>> # add callback / logger specific parameters
>>> parser = callbacks.TransformersModelCheckpointCallback.add_callback_specific_args(parser)
>>> ...
>>> hparams = parser.parse_args()
```

The checkpointing can be controlled in 3 ways with the parameters `--checkpoint_interval`, `--no_epoch_checkpointing` and `--no_val_checkpointing`. The first requires an integer to say the amount of steps between each checkpoints while the second and the third say whether checkpointing at the end of each validation and each epoch should be avoided.

For example, to checkpoint every `10000` steps and to avoid chechpoints at the end of each epoch / validation, do:
```
python main.py ... <training args> ... --checkpoint_interval 10000 --no_epoch_checkpointing --no_val_checkpointing
```

This callbacks searches for 3 attributes of the LightningModule to do a checkpoint. For each one, if it is present it will try to call `save_pretrained`:

- `pl_module.config`: if this attribute is present, the actual configuration is saved;
- `pl_module.model`: if this attribute is present, the actual model is saved;
- `pl_module.tokenizer`: if this attribute is present, the actual tokenizer is saved;

By default, checkpoints will be save to the `outputs/pre_trained_models/<name>` folder.