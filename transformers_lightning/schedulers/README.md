# Schedulers

Schedulers modify the learning rate through the epochs.


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
