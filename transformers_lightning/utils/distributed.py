import torch

def get_total_devices(trainer):
    """
    Compute total number of devices on which training is being performed 
    """
    if trainer.use_dp:
        return 1
    if trainer.use_ddp or trainer.use_ddp2:
        return torch.distributed.get_world_size()
    if trainer.on_gpu:
        return 1
    if trainer.on_tpu:
        return len(trainer.tpu_cores) * trainer.num_nodes
    if trainer.distributed_backend == 'ddp_cpu':
        return trainer.num_processes * trainer.num_nodes
    return 1
