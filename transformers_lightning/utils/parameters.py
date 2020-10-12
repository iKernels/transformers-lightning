"""
Fix some parameters issue
"""

def hacks(hparams, datamodule):
    """
    Implement all hacks necessary.
    """
    
    """
    # no sanity check if no validation required
    if not hasattr(datamodule, 'val_dataloader'):
        hparams.num_sanity_val_steps = 0
        hparams.limit_val_batches = 0.0

    if not hasattr(datamodule, 'test_dataloader'):
        hparams.limit_test_batches = 0.0
    """

    """ Compute total number of steps if not specified. They are required by eventual schedulers. """
    """
    if not hasattr(self, 'max_steps') or self.hparams.max_steps is None:

        assert self.train_config is not None, f"Cannot fix steps without `train_config` defined."

        # retrieve total number of devices
        if self.trainer.on_gpu:
            total_devices = self.trainer.num_nodes * self.trainer.num_processes
        elif self.trainer.on_tpu:
            total_devices = len(self.trainer.tpu_cores) * self.trainer.num_nodes
        elif self.trainer.distributed_backend == 'ddp_cpu':
            total_devices = self.trainer.num_processes
        else:
            total_devices = 1

        # dataset_length retrieval without 
        dataset_length = SuperTransformersDataset.get_length_without_loading(self.train_config, self.hparams)        

        # total number of training batches
        num_training_batches = math.ceil(dataset_length / self.hparams.batch_size)

        # number of batches per epoch
        training_batches_per_epoch = num_training_batches // total_devices

        # total number of steps
        max_steps = (self.hparams.max_epochs * training_batches_per_epoch) // self.hparams.accumulate_grad_batches
        # steps per epoch
        steps_per_epoch = training_batches_per_epoch // self.hparams.accumulate_grad_batches

        self.hparams.max_steps = max_steps
        self.hparams.steps_per_epoch = steps_per_epoch

    if (
        hasattr(self.hparams, "val_check_interval") and
        isinstance(self.hparams.val_check_interval, float) and
        0 <= self.hparams.val_check_interval < 1.0 and
        self.hparams.ds_type == 'iter'
    ):
        fixed_val_check_interval = int(self.hparams.val_check_interval * self.hparams.steps_per_epoch)

        self.hparams.val_check_interval = fixed_val_check_interval
        self.trainer.val_check_interval = fixed_val_check_interval
    """
