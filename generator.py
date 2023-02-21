import math, traceback
import numpy as np, torch
from multiprocessing import shared_memory
from torch.multiprocessing import Process, Pipe
from model import Model, obs_to_torch
from game import kTensorDim, Worker

class DataGenerator:
    def __init__(self, name, model, n_workers, env_per_worker, worker_steps, game_params):
        self.model = model
        self.n_workers = n_workers
        self.env_per_worker = env_per_worker
        self.envs = n_workers * env_per_worker
        self.worker_steps = worker_steps
        self.gamma, self.lamda = game_params[-2:]
        self.device = next(self.model.parameters()).device
        self.total_games = 0
        # TODO: Add networks

        # initialize tensors for observations
        shapes = [(self.envs, *kTensorDim),
                  (self.envs, worker_steps, 2),
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

        self.set_params(game_params)
        self.obs = obs_to_torch(self.obs_np, self.device)
        self.logfile = open('logs/{}/logs'.format(name), 'a')

    def w_range(self, x): return slice(x * self.env_per_worker, (x + 1) * self.env_per_worker)

    def update_model(self, state_dict, epoch = 0):
        target_device = next(self.model.parameters()).device
        for i in state_dict:
            if state_dict[i].device != target_device:
                state_dict[i] = state_dict[i].to(target_device)
        self.model.load_state_dict(state_dict)

        stats = np.zeros((4, 10), dtype='uint64')
        for i in self.workers:
            i.child.send(('get_stats', epoch))
            stats += i.child.recv()
        names = ['SGL', 'DBL', 'TRP', 'TET']
        print(epoch, end='', file=self.logfile)
        for i in range(4):
            print(' ' + names[i], end='', file=self.logfile)
            for j in stats[i]:
                print(' ' + str(j), end='', file=self.logfile)
        print(file=self.logfile, flush=True)

    def set_params(self, game_params):
        for i in self.workers:
            i.child.send(('set_param', game_params[:-2]))
        self.gamma, self.lamda = game_params[-2:]

    def sample(self, train = True, gpu = False, epoch = 0):
        """### Sample data with current policy"""
        actions = torch.zeros((self.worker_steps, self.envs), dtype = torch.int32, device = self.device)
        obs = torch.zeros((self.worker_steps, self.envs, *kTensorDim), dtype = torch.float32, device = self.device)
        log_pis = torch.zeros((self.worker_steps, self.envs), dtype = torch.float32, device = self.device)
        values = torch.zeros((self.worker_steps, 2, self.envs), dtype = torch.float32, device = self.device)

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
                values[t] = v[:2] # remove stdev
                a = pi.sample()
                actions[t] = a
                log_pis[t] = pi.log_prob(a)
                actions_cpu = a.cpu().numpy()

            # run sampled actions on each worker
            # workers will place results in self.obs_np,rewards,done
            for w, worker in enumerate(self.workers):
                worker.child.send(('step', (t, actions_cpu[self.w_range(w)], epoch)))
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
        score_max = self.rewards[:,:,1].max()
        if train:
            ret_info['maxk'].append(score_max / 1e-2)
            ret_info['mil_games'].append(self.total_games * 1e-6)
            if tot_lines > 0:
                ret_info['perline'].append(tot_score * 1e-3 / tot_lines)

            # calculate advantages
            advantages, raw_devs = self._calc_advantages(self.done, self.rewards, values)
            samples = {
                'obs': obs,
                'actions': actions,
                'log_pis': log_pis,
                'raw_devs': raw_devs,
                'values': values.transpose(0, 1).reshape(1, 2, -1),
                'advantages': advantages.transpose(0, 1).reshape(1, 2, -1),
            }
        else:
            samples = {
                'obs': obs,
                'log_pis': log_pis,
                'values': values.transpose(0, 1).reshape(1, 2, -1),
            }

        # samples are currently in [time, workers] table, flatten it
        # for values & advantages, this just flattens the first two dimensions
        for i in samples:
            samples[i] = samples[i].reshape(-1, *samples[i].shape[2:])

        if not gpu:
            for i in samples:
                samples[i] = samples[i].cpu()
        for i in list(ret_info):
            if ret_info[i]:
                ret_info[i] = np.mean(ret_info[i])
            else:
                del ret_info[i]
        return samples, ret_info

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: torch.Tensor) -> torch.Tensor:
        """### Calculate advantages"""
        # TODO: generate intrinsic rewards
        # TODO: 0(value), 1(raw), 2(dev) -> 0(value), 1(intrinsic), 2(raw), 3(dev)
        with torch.no_grad():
            rewards = torch.permute(torch.from_numpy(rewards).to(self.device), (1, 2, 0))
            done_neg = ~torch.transpose(torch.from_numpy(done).to(self.device), 0, 1)
            # done_neg repeat and set [:,1] to 1

            # advantages table
            advantages = torch.zeros((self.worker_steps, 2, self.envs), dtype = torch.float32, device = self.device)
            raw_devs = torch.zeros((self.worker_steps, self.envs), dtype = torch.float32, device = self.device)
            last_advantage = torch.zeros((2, self.envs), dtype = torch.float32, device = self.device)

            # $V(s_{t+1})$
            _, last_value = self.model(self.obs)
            last_dev = last_value[2]
            last_value = last_value[:2] # remove stdev
            gammas = torch.Tensor([self.gamma, 1.0]).unsqueeze(1).to(self.device)
            lamdas = torch.Tensor([self.lamda, 1.0]).unsqueeze(1).to(self.device)
            # last_dev = last_value[3]
            # last_value = last_value[:3] # remove stdev
            # gammas = torch.Tensor([self.gamma, self.gamma_int, 1.0]).unsqueeze(1).to(self.device)
            # lamdas = torch.Tensor([self.lamda, self.lamda, 1.0]).unsqueeze(1).to(self.device)

            for t in reversed(range(self.worker_steps)):
                # mask if episode completed after step $t$
                mask = done_neg[t]
                last_dev *= mask # done_neg[t,0]
                # last_value = last_value * mask
                # last_advantage = last_advantage * mask
                # $\delta_t = reward[t] - value[t] + last_value * gammas$
                # $\hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}} (gam * lam * last_advantage)$
                last_advantage = rewards[t] - values[t] + gammas * (last_value + lamdas * last_advantage) * mask # done_neg[t]
                # note that we are collecting in reverse order.
                advantages[t] = last_advantage
                raw_devs[t] = last_dev
                last_value = values[t]
            return advantages, raw_devs

    def destroy(self):
        try:
            for i in self.workers: i.child.send(('close', None))
        except: pass
        for i in self.shms:
            i.close()
            i.unlink()

def generator_process(remote, name, model_args, *args):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = Model(*model_args).to(device)
        generator = DataGenerator(name, model, *args)
        samples = None
        while True:
            cmd, data = remote.recv()
            if cmd == "update_model":
                generator.update_model(data[0], data[1])
            elif cmd == "set_param":
                generator.set_params(data)
            elif cmd == "start_generate":
                samples = generator.sample(epoch = data)
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
    def __init__(self, name, model, c, game_params, device):
        self.device = device
        self.child, parent = Pipe()
        ctx = torch.multiprocessing.get_context('spawn')
        self.process = ctx.Process(target = generator_process,
                args = (parent, name, c.model_args(), c.n_workers, c.env_per_worker, c.worker_steps, game_params))
        self.process.start()
        self.SendModel(model, -1)

    def SendModel(self, model, epoch):
        state_dict = model.state_dict()
        for i in state_dict:
            state_dict[i] = state_dict[i].cpu()
        self.child.send(('update_model', (state_dict, epoch)))

    def StartGenerate(self, epoch):
        self.child.send(('start_generate', epoch))

    def SetParams(self, game_params):
        self.child.send(('set_param', game_params))

    def GetData(self):
        self.child.send(('get_data', None))
        data, info = self.child.recv()
        for i in data:
            data[i] = data[i].to(self.device)
        return data, info

    def Close(self):
        self.child.send(('close', None))
