#!/usr/bin/env python3

# Modified from https://github.com/vpj/rl_samples

import sys, traceback, os, collections, math, time
from typing import Dict, List
from sortedcontainers import SortedList
from multiprocessing import shared_memory

import numpy as np, torch
from torch import optim
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler

from labml import monit, tracker, logger, experiment
from labml.configs import FloatDynamicHyperParam

from game import Worker, kTensorDim
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
        self.total_games = 0

        # model for sampling
        self.model = Model(c.channels, c.blocks).to(device)

        # dynamic hyperparams
        self.cur_lr = self.c.lr()
        self.cur_reg_l2 = self.c.reg_l2()
        self.cur_pre_trans = 0.
        self.cur_right_gain = 0.
        self.cur_neg_mul = 0.
        self.cur_entropy_weight = self.c.entropy_weight()
        self.cur_gamma = self.c.gamma()
        self.cur_lamda = self.c.lamda()

        # optimizer
        self.scaler = GradScaler()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.cur_lr,
                weight_decay = self.cur_reg_l2)

        # initialize tensors for observations
        shapes = [(self.envs, *kTensorDim),
                  (self.envs, self.c.worker_steps),
                  (self.envs, self.c.worker_steps)]
        types = [np.dtype('float32'), np.dtype('float32'), np.dtype('bool')]
        self.shms = [
            shared_memory.SharedMemory(create = True, size = math.prod(shape) * typ.itemsize)
            for shape, typ in zip(shapes, types)
        ]
        self.obs_np, self.rewards, self.done = [
            np.ndarray(shape, dtype = typ, buffer = shm.buf)
            for shm, shape, typ in zip(self.shms, shapes, types)
        ]
        # create workers
        shm = [(shm.name, shape, typ) for shm, shape, typ in zip(self.shms, shapes, types)]
        self.workers = [Worker(name, shm, self.w_range(i), 27 + i) for i in range(self.c.n_workers)]
        self.set_game_param(self.c.pre_trans(), self.c.right_gain(), self.c.neg_mul())
        for i in self.workers: i.child.send(('reset', None))
        for i in self.workers: i.child.recv()

        self.obs = obs_to_torch(self.obs_np, device)

    def set_optim(self, lr, reg_l2):
        if lr == self.cur_lr and reg_l2 == self.cur_reg_l2: return
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = reg_l2
        self.cur_lr = lr
        self.cur_reg_l2 = reg_l2

    def set_game_param(self, pre_trans, gain, neg_mul):
        if pre_trans == self.cur_pre_trans and gain == self.cur_right_gain and \
                neg_mul == self.cur_neg_mul: return
        for i in self.workers: i.child.send(('set_param', (pre_trans, gain, neg_mul)))
        self.cur_pre_trans = pre_trans
        self.cur_right_gain = gain
        self.cur_neg_mul = neg_mul

    def set_weight_param(self, entropy, gamma, lamda):
        self.cur_entropy_weight = entropy
        self.cur_gamma = gamma
        self.cur_lamda = lamda

    def w_range(self, x): return slice(x * self.c.env_per_worker, (x + 1) * self.c.env_per_worker)

    def destroy(self):
        try:
            for i in self.workers: i.child.send(('close', None))
        except: pass
        for i in self.shms:
            i.close()
            i.unlink()
        self.workers = None
        self.shms = None
        self.obs_np = None
        self.rewards = None
        self.done = None

    def sample(self, train = True) -> (Dict[str, np.ndarray], List):
        """### Sample data with current policy"""
        actions = torch.zeros((self.c.worker_steps, self.envs), dtype = torch.int32, device = device)
        obs = torch.zeros((self.c.worker_steps, self.envs, *kTensorDim), dtype = torch.float32, device = device)
        log_pis = torch.zeros((self.c.worker_steps, self.envs), dtype = torch.float32, device = device)
        values = torch.zeros((self.c.worker_steps, self.envs), dtype = torch.float32, device = device)

        # sample `worker_steps` from each worker
        tot_lines = 0
        tot_score = 0
        for t in range(self.c.worker_steps):
            with torch.no_grad():
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[t] = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$
                pi, v = self.model(self.obs)
                values[t] = v
                a = pi.sample()
                actions[t] = a
                log_pis[t] = pi.log_prob(a)
                actions_cpu = a.cpu().numpy()

            # run sampled actions on each worker
            # workers will place results in self.obs_np,rewards,done
            for w, worker in enumerate(self.workers):
                worker.child.send(('step', (t, actions_cpu[self.w_range(w)], tracker.get_global_step())))
            for i in self.workers:
                info_arr = i.child.recv()
                # collect episode info, which is available if an episode finished
                if train:
                    self.total_games += len(info_arr)
                    for info in info_arr:
                        tot_lines += info['lines']
                        tot_score += info['score']
                        tracker.add('reward', info['reward'])
                        tracker.add('scorek', info['score'] * 1e-3)
                        tracker.add('lns', info['lines'])
                        tracker.add('len', info['length'])
                        tracker.add('trt', info['tetris'])
                        tracker.add('rtrt', info['rtetris'])
            self.obs = obs_to_torch(self.obs_np, device)

        # reshape rewards & log rewards
        reward_max = self.rewards.max()
        if train:
            tracker.add('maxk', reward_max / 1e-2)
            tracker.add('mil_games', self.total_games * 1e-6)
            tracker.add('perline', tot_score * 1e-3 / tot_lines)

        # calculate advantages
        advantages = self._calc_advantages(self.done, self.rewards, values)
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'log_pis': log_pis,
            'advantages': advantages
        }
        # samples are currently in [time, workers] table, flatten it
        for i in samples:
            samples[i] = samples[i].reshape(-1, *samples[i].shape[2:])
        return samples

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: torch.Tensor) -> torch.Tensor:
        """### Calculate advantages"""
        with torch.no_grad():
            rewards = torch.transpose(torch.from_numpy(rewards).to(device), 0, 1)
            done_neg = ~torch.transpose(torch.from_numpy(done).to(device), 0, 1)

            # advantages table
            advantages = torch.zeros((self.c.worker_steps, self.envs), dtype = torch.float32, device = device)
            last_advantage = torch.zeros(self.envs, dtype = torch.float32, device = device)

            # $V(s_{t+1})$
            _, last_value = self.model(self.obs)

            for t in reversed(range(self.c.worker_steps)):
                # mask if episode completed after step $t$
                mask = done_neg[t]
                last_value = last_value * mask
                last_advantage = last_advantage * mask
                # $\delta_t$
                delta = rewards[t] + self.cur_gamma * last_value - values[t]
                # $\hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}}$
                last_advantage = delta + self.cur_gamma * self.cur_lamda * last_advantage
                # note that we are collecting in reverse order.
                advantages[t] = last_advantage
                last_value = values[t]
            return advantages

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
        samples['advantages'] = Main._normalize(samples['advantages'])

    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range: float) -> torch.Tensor:
        """## PPO Loss"""
        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        pi_val, value = self.model(samples['obs'], False)
        pi = Categorical(logits = pi_val)

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

        # we want to maximize $\mathcal{L}^{CLIP+VF+EB}(\theta)$
        # so we take the negative of it as the loss
        loss = -(policy_reward - self.c.vf_weight * vf_loss + \
                self.cur_entropy_weight * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()
        tracker.add({'policy_reward': policy_reward,
                     'vf_loss': vf_loss ** 0.5,
                     'entropy_bonus': entropy_bonus,
                     'kl_div': approx_kl_divergence,
                     'clip_fraction': clip_fraction})
        return loss

    def run_training_loop(self):
        """### Run training loop"""
        offset = tracker.get_global_step()
        if offset > 100:
            # If resumed, sample several iterations first to reduce sampling bias
            for i in range(16): self.sample(False)
            tracker.save() # increment step
        for _ in monit.loop(self.c.updates - offset):
            update = tracker.get_global_step()
            # sample with current policy
            samples = self.sample()
            # train the model
            self.train(samples)
            # write summary info to the writer, and log to the screen
            tracker.save()
            if (update + 1) % 2 == 0:
                self.set_optim(self.c.lr(), self.c.reg_l2())
                self.set_game_param(self.c.pre_trans(), self.c.right_gain(), self.c.neg_mul())
                self.set_weight_param(self.c.entropy_weight(), self.c.gamma(), self.c.lamda())
            if (update + 1) % 25 == 0: logger.log()
            if (update + 1) % 250 == 0: experiment.save_checkpoint()

import argparse

if __name__ == "__main__":
    conf, args, _ = LoadConfig()
    m = Main(conf, args['name'])
    experiment.add_model_savers({
            'model': TorchSaver('model', m.model),
            'scaler': TorchSaver('scaler', m.scaler),
            'optimizer': TorchSaver('optimizer', m.optimizer),
        })
    if len(args['uuid']): experiment.load(args['uuid'], args['checkpoint'])
    with experiment.start():
        try: m.run_training_loop()
        except Exception as e: print(traceback.format_exc())
        finally: m.destroy()
