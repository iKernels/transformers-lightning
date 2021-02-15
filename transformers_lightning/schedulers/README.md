# Schedulers

Schedulers modify the learning rate through the epochs. *Every* scheduler is designed to be called every *step* and *not* every *epoch*.


## LinearScheduler

This scheduler will linearly decrease the learning rate from its default value `learning_rate` to `0` during the `num_training_steps`.


## LinearSchedulerWithWarmup

This scheduler will linearly increment the learning rate from 0 to the `learning_rate` in the first `num_warmup_steps` steps. After that, it will linearly decrease it up to `num_training_steps`, that is, the total number of training steps.


## ConstantScheduler

This scheduler will always return the original `learning_rate`.


## ConstantSchedulerWithWarmup

This scheduler will linearly increment the learning rate from 0 to the `learning_rate` in the first `num_warmup_steps` steps. After that, it will always return the original `learning_rate`.


## CosineSchedulerWithWarmup

This scheduler will linearly increment the learning rate from 0 to the `learning_rate` in the first `num_warmup_steps` steps. After that, it will decrease it up to `0` following the value of the cosine function.


## CosineSchedulerWithWarmupAndHardRestart

This scheduler will linearly increment the learning rate from 0 to the `learning_rate` in the first `num_warmup_steps` steps. After that, it will decrease it up to `0` following the value of the cosine function with several hard restarts.


## PolynomialLayerwiseDecaySchedulerWithWarmup

Create a polynomially decreasing scheduler. Conversion of `https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay`. More informations about the default parameters can be found on the documentation of `_LRScheduler` in the `torch` project.

The `layerwise_lr_decay_power` parameter allows to multiply the learning rate for each group by a factor `f = layerwise_lr_decay_power^(max(depth) - depth)`. `depth` is a key that should be defined in every group of parameters, along with the usual `weight_decay`. A small `depth` will mean that the learning rate will be only slightly decreased. A high `depth` will reduce a lot the effects of the training on that parameters.
