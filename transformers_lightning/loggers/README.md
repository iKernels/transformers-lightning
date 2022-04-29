# Loggers

Loggers automatically save metrics, hyperparameters and more to disk while training. This package contains now only a new logger to the [jsonboard](https://github.com/lucadiliello/jsonboard) project. `jsonboard` is a convinient alternative to `tensorboard` which is based on `json` files instead of `tf.events`. This enables an easier modification and inspection of log files, which is one of the main drawbacks of `tensorboard`.

You may use the logger by simply adding it to the Trainer.

```python
from pytorch_lightning import Trainer
from transformers_lightning.loggers import JsonBoardLogger

logger = JsonBoardLogger("js_logs", name="experiment-1")
trainer = Trainer(logger=logger)
```
