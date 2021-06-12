import argparse

from labml import experiment
from labml.configs import BaseConfigs, FloatDynamicHyperParam

class Configs(BaseConfigs):
    # #### Configurations
    # $\gamma$ and $\lambda$ for advantage calculation
    gamma: float = FloatDynamicHyperParam(0.99, range_ = (0.98, 1))
    lamda: float = FloatDynamicHyperParam(0.95, range_ = (0.9, 1))
    # number of updates
    updates: int = 100000
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
    channels: int = 160
    blocks: int = 10
    lr: float = FloatDynamicHyperParam(1e-4, range_ = (0, 1e-3))
    clipping_range: float = 0.2
    vf_weight: float = 0.5
    target_prob_weight: float = FloatDynamicHyperParam(0.8, range_ = (0, 2))
    entropy_weight: float = FloatDynamicHyperParam(3e-2, range_ = (0, 5e-2))
    prob_reg_weight: float = FloatDynamicHyperParam(2e-4, range_ = (0, 1e-2))
    reg_l2: float = FloatDynamicHyperParam(0., range_ = (0, 5e-5))

    right_gain: float = FloatDynamicHyperParam(0.501, range_ = (0, 1))
    fix_prob: float = FloatDynamicHyperParam(1., range_ = (0, 1))
    neg_mul: float = FloatDynamicHyperParam(0., range_ = (0, 1))
    step_reward: float = FloatDynamicHyperParam(2e-5, range_ = (0, 2e-5))

def LoadConfig(with_experiment = True):
    parser = argparse.ArgumentParser()
    if with_experiment:
        parser.add_argument('name')
        parser.add_argument('uuid', nargs = '?', default = '')
        parser.add_argument('checkpoint', nargs = '?', type = int, default = None)
    conf = Configs()
    keys = conf._to_json()
    dynamic_keys = set()
    for key in keys:
        ptype = type(conf.__getattribute__(key))
        if ptype == FloatDynamicHyperParam:
            ptype = float
            dynamic_keys.add(key)
        parser.add_argument('--' + key.replace('_', '-'), type = ptype)

    args, others = parser.parse_known_args()
    args = vars(args)
    override_dict = {}
    for key in keys:
        if key not in dynamic_keys and args[key] is not None: override_dict[key] = args[key]
    conf = Configs()
    for key in dynamic_keys:
        if args[key] is not None:
            conf.__getattribute__(key).set_value(args[key])
    if with_experiment:
        try:
            if len(args['name']) == 32:
                int(args['name'], 16)
                parser.error('Experiment name should not be uuid-like')
        except ValueError: pass
        os.makedirs('logs/{}'.format(args['name']), exist_ok = True)
        experiment.create(name = args['name'])
        experiment.configs(conf, override_dict)
    else:
        for key, val in override_dict:
            conf.__setattr__(key, val)
    return conf, others
