# Schedulers

Schedulers modify the learning rate through the epochs.

## LinearSchedulerWithWarmup

This scheduler will linearly increment the learning rate from 0 to the `learning_rate` in the first `num_warmup_steps` steps. After that, it will linearly decrease it up to `num_training_steps`, that is, the total number of training steps.
