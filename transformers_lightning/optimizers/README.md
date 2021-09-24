# Optimizers

Optimizers manage weight update starting from gradient values. They may have complex internal states to better move on the loss multi-dimensional surface. Please use the fixed signature `__init__(hyperparameters: Namespace, named_parameters: Generator) -> None` for all the subclasses. 


## ElectraAdamW

This optimizer is same as `AdamW` but for a small fix to the moving average update mechanism. Original implementation can be found [here](https://github.com/google-research/electra/blob/f93f3f81cdc13435dd3e85766852d00ff3e00ab5/model/optimization.py#L70).
