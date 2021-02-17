from labml.configs import BaseConfigs

class Configs(BaseConfigs):
    # #### Configurations
    # $\gamma$ and $\lambda$ for advantage calculation
    gamma: float = 0.9994
    lamda: float = 0.95
    # number of updates
    updates: int = 40000
    # number of epochs to train the model with sampled data
    epochs: int = 1
    # number of worker processes
    n_workers: int = 16
    env_per_worker: int = 8
    # number of steps to run on each process for a single update
    worker_steps: int = 256
    # size of mini batches
    n_update_per_epoch: int = 32
    mini_batch_size: int = 512
    channels: int = 128
    blocks: int = 10
    lr: float = 1e-4
    clipping_range: float = 0.2
    vf_weight: float = 0.5
    entropy_weight: float = 1e-2
    reg_l2: float = 1e-5
