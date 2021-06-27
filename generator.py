import math, traceback
import numpy as np, torch
from multiprocessing import shared_memory
from torch.multiprocessing import Process, Pipe
from model import Model, obs_to_torch
from game import kTensorDim, Worker

class DataGenerator:
    def __init__(self, name, model, n_workers, env_per_worker, worker_steps, pre_trans, right_gain, neg_mul, gamma, lamda):
        self.model = model
        self.n_workers = n_workers
        self.env_per_worker = env_per_worker
        self.envs = n_workers * env_per_worker
        self.worker_steps = worker_steps
        self.gamma = gamma
        self.lamda = lamda
        self.device = next(self.model.parameters()).device
        self.total_games = 0

        # initialize tensors for observations
        shapes = [(self.envs, *kTensorDim),
                  (self.envs, worker_steps),
                  (self.envs, worker_steps)]
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
        self.workers = [Worker(name, shm, self.w_range(i), 27 + i) for i in range(self.n_workers)]
        for i in self.workers: i.child.send(('reset', None))
        for i in self.workers: i.child.recv()

        self.set_params(pre_trans, right_gain, neg_mul, gamma, lamda)
        self.obs = obs_to_torch(self.obs_np, self.device)

    def w_range(self, x): return slice(x * self.env_per_worker, (x + 1) * self.env_per_worker)

    def update_model(self, state_dict):
        target_device = next(self.model.parameters()).device
        for i in state_dict:
            if state_dict[i].device != target_device:
                state_dict[i] = state_dict[i].to(target_device)
        self.model.load_state_dict(state_dict)

    def set_params(self, pre_trans, right_gain, neg_mul, gamma, lamda):
        for i in self.workers:
            i.child.send(('set_param', (pre_trans, right_gain, neg_mul)))
        self.gamma = gamma
        self.lamda = lamda

    def sample(self, train = True, gpu = False, step = 0):
        """### Sample data with current policy"""
        actions = torch.zeros((self.worker_steps, self.envs), dtype = torch.int32, device = self.device)
        obs = torch.zeros((self.worker_steps, self.envs, *kTensorDim), dtype = torch.float32, device = self.device)
        log_pis = torch.zeros((self.worker_steps, self.envs), dtype = torch.float32, device = self.device)
        values = torch.zeros((self.worker_steps, self.envs), dtype = torch.float32, device = self.device)

        if train:
            ret_info = {
                'reward': [],
                'scorek': [],
                'lns': [],
                'len': [],
                'trt': [],
                'rtrt': [],
                'maxk': [],
                'mil_games': [],
                'perline': [],
            }
        else:
            ret_info = {}

        # sample `worker_steps` from each worker
        tot_lines = 0
        tot_score = 0
        for t in range(self.worker_steps):
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
                worker.child.send(('step', (t, actions_cpu[self.w_range(w)], step)))
            for i in self.workers:
                info_arr = i.child.recv()
                # collect episode info, which is available if an episode finished
                if train:
                    self.total_games += len(info_arr)
                    for info in info_arr:
                        tot_lines += info['lines']
                        tot_score += info['score']
                        ret_info['reward'].append(info['reward'])
                        ret_info['scorek'].append(info['score'] * 1e-3)
                        ret_info['lns'].append(info['lines'])
                        ret_info['len'].append(info['length'])
                        ret_info['trt'].append(info['tetris'])
                        ret_info['rtrt'].append(info['rtetris'])
            self.obs = obs_to_torch(self.obs_np, self.device)

        # reshape rewards & log rewards
        reward_max = self.rewards.max()
        if train:
            ret_info['maxk'].append(reward_max / 1e-2)
            ret_info['mil_games'].append(self.total_games * 1e-6)
            ret_info['perline'].append(tot_score * 1e-3 / tot_lines)

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
            if not gpu:
                samples[i] = samples[i].cpu()
        for i in ret_info:
            ret_info[i] = np.mean(ret_info[i])
        return samples, ret_info

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: torch.Tensor) -> torch.Tensor:
        """### Calculate advantages"""
        with torch.no_grad():
            rewards = torch.transpose(torch.from_numpy(rewards).to(self.device), 0, 1)
            done_neg = ~torch.transpose(torch.from_numpy(done).to(self.device), 0, 1)

            # advantages table
            advantages = torch.zeros((self.worker_steps, self.envs), dtype = torch.float32, device = self.device)
            last_advantage = torch.zeros(self.envs, dtype = torch.float32, device = self.device)

            # $V(s_{t+1})$
            _, last_value = self.model(self.obs)

            for t in reversed(range(self.worker_steps)):
                # mask if episode completed after step $t$
                mask = done_neg[t]
                last_value = last_value * mask
                last_advantage = last_advantage * mask
                # $\delta_t$
                delta = rewards[t] + self.gamma * last_value - values[t]
                # $\hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}}$
                last_advantage = delta + self.gamma * self.lamda * last_advantage
                # note that we are collecting in reverse order.
                advantages[t] = last_advantage
                last_value = values[t]
            return advantages

    def destroy(self):
        try:
            for i in self.workers: i.child.send(('close', None))
        except: pass
        for i in self.shms:
            i.close()
            i.unlink()

def generator_process(remote, name, channels, blocks, *args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = Model(channels, blocks).to(device)
        generator = DataGenerator(name, model, *args)
        samples = None
        while True:
            cmd, data = remote.recv()
            if cmd == "update_model":
                generator.update_model(data)
            elif cmd == "set_param":
                generator.set_params(*data)
            elif cmd == "start_generate":
                samples = generator.sample(step = data)
            elif cmd == "get_data":
                remote.send(samples)
                samples = None
            elif cmd == "close":
                return
            else:
                raise NotImplementedError
    except:
        print(traceback.format_exc())
        raise
    finally:
        remote.close()

class GeneratorProcess:
    def __init__(self, name, model, c):
        self.child, parent = Pipe()
        ctx = torch.multiprocessing.get_context('spawn')
        self.process = ctx.Process(target = generator_process,
                args = (parent, name, c.channels, c.blocks, c.n_workers, c.env_per_worker,
                    c.worker_steps, c.pre_trans(), c.right_gain(), c.neg_mul(), c.gamma(), c.lamda()))
        self.process.start()
        self.SendModel(model)

    def SendModel(self, model):
        state_dict = model.state_dict()
        for i in state_dict:
            state_dict[i] = state_dict[i].cpu()
        self.child.send(('update_model', state_dict))

    def StartGenerate(self, step):
        self.child.send(('start_generate', step))

    def SetParams(self, pre_trans, right_gain, neg_mul, gamma, lamda):
        self.child.send(('set_param', (pre_trans, right_gain, neg_mul, gamma, lamda)))

    def GetData(self):
        self.child.send(('get_data', None))
        data, info = self.child.recv()
        for i in data:
            data[i] = data[i].cuda()
        return data, info

    def Close(self):
        self.child.send(('close', None))
