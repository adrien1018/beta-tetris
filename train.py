#!/usr/bin/env python3

# Modified from https://github.com/vpj/rl_samples

import sys, traceback, os, collections, math, time
from typing import Dict, List

import numpy as np, torch
from torch import optim
from torch.distributions import Categorical, Normal, kl_divergence
from torch.cuda.amp import autocast, GradScaler

import labml.lab
from labml import monit, tracker, logger, experiment
from labml.configs import FloatDynamicHyperParam

from game import kTensorDim
from generator import GeneratorProcess
from model import Model, obs_to_torch
from config import Configs, LoadConfig
from saver import TorchSaver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

class Main:
    def __init__(self, c: Configs, name: str):
        self.name = name
        self.c = c
        # total number of samples for a single update
        self.envs = self.c.n_workers * self.c.env_per_worker
        self.batch_size = self.envs * self.c.worker_steps
        assert (self.batch_size % (self.c.n_update_per_epoch * self.c.mini_batch_size) == 0)
        self.update_batch_size = self.batch_size // self.c.n_update_per_epoch

        # #### Initialize
        # model for sampling
        self.model = Model(c.channels, c.blocks).to(device)

        # dynamic hyperparams
        self.cur_lr = self.c.lr()
        self.cur_reg_l2 = self.c.reg_l2()
        self.cur_game_params = (0., 0., 0., 0., 0., 0., 0.)
        self.cur_entropy_weight = self.c.entropy_weight()
        self.cur_raw_weight = self.c.raw_weight()

        # optimizer
        self.scaler = GradScaler()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.cur_lr,
                weight_decay = self.cur_reg_l2)

        # generator
        cur_params = self.get_game_params()
        self.generator = GeneratorProcess(self.name, self.model, self.c, cur_params, device)
        self.set_game_params(cur_params)

    def get_game_params(self):
        return (self.c.pre_trans(), self.c.left_deduct(), self.c.neg_mul(), self.c.reward_ratio(),
                self.c.normal_rate(), self.c.gamma(), self.c.lamda())

    def set_optim(self, lr, reg_l2):
        if lr == self.cur_lr and reg_l2 == self.cur_reg_l2: return
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = reg_l2
        self.cur_lr = lr
        self.cur_reg_l2 = reg_l2

    def set_game_params(self, game_params):
        if game_params == self.cur_game_params: return
        self.generator.SetParams(game_params)
        self.cur_game_params = game_params

    def set_weight_param(self, entropy, raw):
        self.cur_entropy_weight = entropy
        self.cur_raw_weight = raw

    def destroy(self):
        self.generator.Close()
        self.generator = None

    def train(self, samples: Dict[str, torch.Tensor]):
        """### Train the model based on samples"""
        self._preprocess_samples(samples)
        for _ in range(self.c.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)
            for start in range(0, self.batch_size, self.update_batch_size):
                # get mini batch
                end = start + self.update_batch_size
                # train
                self.optimizer.zero_grad()
                loss_mul = self.update_batch_size // self.c.mini_batch_size
                for t_start in range(start, end, self.c.mini_batch_size):
                    t_end = t_start + self.c.mini_batch_size
                    mini_batch_indexes = indexes[t_start:t_end]
                    mini_batch = {}
                    with torch.no_grad():
                        for k, v in samples.items():
                            mini_batch[k] = v[t_start:t_end]
                    loss = self._calc_loss(clip_range = self.c.clipping_range,
                                        samples = mini_batch) / loss_mul
                    self.scaler.scale(loss).backward()
                # compute gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 0.5)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 16)
                self.scaler.step(self.optimizer)
                self.scaler.update()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    @staticmethod
    def _preprocess_samples(samples: Dict[str, torch.Tensor]):
        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        samples['returns'] = (samples['values'] + samples['advantages']).float()
        samples['raw_returns'] = samples['returns'][1]
        samples['returns'] = samples['returns'][0]
        samples['values'] = samples['values'][0]
        samples['advantages'] = Main._normalize(samples['advantages'][0])
        samples['raw_devs'].clamp_(min=1e-5)

    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range: float) -> torch.Tensor:
        """## PPO Loss"""
        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        pi_val, value = self.model(samples['obs'], False)
        pi = Categorical(logits = pi_val)
        raw_dist = Normal(value[1], value[2])
        value = value[0]

        # #### Policy
        log_pi = pi.log_prob(samples['actions'])
        # *this is different from rewards* $r_t$.
        ratio = torch.exp(log_pi - samples['log_pis'])
        # The ratio is clipped to be close to 1.
        # Using the normalized advantage
        #  $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$
        #  introduces a bias to the policy gradient estimator,
        #  but it reduces variance a lot.
        clipped_ratio = ratio.clamp(min = 1.0 - clip_range,
                                    max = 1.0 + clip_range)
        # advantages are normalized
        policy_reward = torch.min(ratio * samples['advantages'],
                                  clipped_ratio * samples['advantages'])
        policy_reward = policy_reward.mean()

        # #### Entropy Bonus
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # #### Value
        # Clipping makes sure the value function $V_\theta$ doesn't deviate
        #  significantly from $V_{\theta_{OLD}}$.
        clipped_value = samples['values'] + (value - samples['values']).clamp(
                min = -clip_range, max = clip_range)
        vf_loss = torch.max((value - samples['returns']) ** 2,
                            (clipped_value - samples['returns']) ** 2)
        vf_loss = 0.5 * vf_loss.mean()

        # #### Score distribution
        raw_loss = kl_divergence(Normal(samples['raw_returns'], samples['raw_devs']), raw_dist).mean()

        # we want to maximize $\mathcal{L}^{CLIP+VF+EB}(\theta)$
        # so we take the negative of it as the loss
        loss = -(policy_reward - self.c.vf_weight * vf_loss - self.cur_raw_weight * raw_loss + \
                self.cur_entropy_weight * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()
        tracker.add({'policy_reward': policy_reward,
                     'vf_loss': vf_loss ** 0.5,
                     'raw_loss': raw_loss,
                     'entropy_bonus': entropy_bonus,
                     'kl_div': approx_kl_divergence,
                     'clip_fraction': clip_fraction})
        return loss

    def run_training_loop(self):
        """### Run training loop"""
        offset = tracker.get_global_step()
        if offset > 100:
            # If resumed, sample several iterations first to reduce sampling bias
            self.generator.SendModel(self.model)
            for i in range(16): self.generator.StartGenerate(offset)
            tracker.save() # increment step
        else:
            self.generator.StartGenerate(offset)
        for _ in monit.loop(self.c.updates - offset):
            update = tracker.get_global_step()
            # sample with current policy
            samples, info = self.generator.GetData()
            self.generator.StartGenerate(update)
            tracker.add(info)
            # train the model
            self.train(samples)
            self.generator.SendModel(self.model)
            # write summary info to the writer, and log to the screen
            tracker.save()
            if (update + 1) % 2 == 0:
                self.set_optim(self.c.lr(), self.c.reg_l2())
                self.set_game_params(self.get_game_params())
                self.set_weight_param(self.c.entropy_weight(), self.c.raw_weight())
            if (update + 1) % 25 == 0: logger.log()
            if (update + 1) % self.c.save_interval == 0: experiment.save_checkpoint()


def claim_experiment(uuid: str):
    from urllib.request import Request, urlopen
    from urllib.parse import urlsplit, urlunsplit
    url = labml.lab.get_info()['configs']['web_api']
    scheme, host, _, _, _ = urlsplit(url)
    url = urlunsplit((scheme, host, f'/api/v1/run/{uuid}/claim', '', ''))
    urlopen(Request(url, method='PUT'))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    conf, args, _ = LoadConfig()
    m = Main(conf, args['name'])
    experiment.add_model_savers({
            'model': TorchSaver('model', m.model),
            'scaler': TorchSaver('scaler', m.scaler),
            'optimizer': TorchSaver('optimizer', m.optimizer, not args['ignore_optimizer']),
        })
    if len(args['uuid']):
        try:
            uuid = '{}-{:03d}'.format(args['name'], int(args['uuid']))
        except ValueError:
            uuid = args['uuid']
            if uuid == 'last':
                nd = 1
                while os.path.exists('logs/{0}/{0}-{1:03d}'.format(args['name'], nd)):
                    nd += 1
                if nd > 1:
                    uuid = '{}-{:03d}'.format(args['name'], nd - 1)
        experiment.load(uuid, args['checkpoint'])
    with experiment.start():
        claim_experiment(experiment.get_uuid())
        try: m.run_training_loop()
        except Exception as e: print(traceback.format_exc())
        finally: m.destroy()
