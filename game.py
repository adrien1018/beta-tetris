import hashlib, traceback
import numpy as np, torch
from multiprocessing import shared_memory
from torch.multiprocessing import Process, Pipe

import tetris

kTensorDim = tetris.Tetris.StateShape()
kH, kW = kTensorDim[1:]

class Game:
    def __init__(self, seed: int):
        self.args = (0, False)
        self.env = tetris.Tetris(seed)
        # pre_trans, left_deduct, penalty_multiplier, reward_ratio, normal_rate
        self.reset_params = (1., 0., 0., 1., 1.)
        self.reset()

    def step(self, action):
        r, x, y = action // 200, action // 10 % 20, action % 10
        reward = self.env.InputPlacement(r, x, y)
        self.reward += reward[0]
        self.length += 0.5

        info = None
        is_over = False
        if self.env.IsOver():
            trt, rtrt = self.env.GetTetrisStat()
            info = {'reward': self.reward,
                    'score': self.env.GetScore(),
                    'lines': self.env.GetLines(),
                    'tetris': trt,
                    'rtetris': rtrt,
                    'length': self.length}
            is_over = True
            self.reset()
        return self.env.GetState(), reward, is_over, info

    def reset(self):
        self.reward = 0.
        self.length = 0.
        self.env.ResetRandom(*self.reset_params)
        return self.env.GetState()

def worker_process(remote, name: str, shms: list, idx: slice, seed: int):
    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    shm_obs, shm_reward, shm_over = [
            np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    # create game environments
    num = idx.stop - idx.start
    Seed = lambda x: int.from_bytes(hashlib.sha256(
        int.to_bytes(seed, 4, 'little') + int.to_bytes(x, 4, 'little')).digest(), 'little')
    games = [Game(Seed(i)) for i in range(num)]
    # wait for instructions from the connection and execute them
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                step, data, epoch = data
                result = []
                for i in range(num):
                    result.append(games[i].step(data[i]))
                obs, reward, over, info = zip(*result)
                shm_obs[idx] = np.stack(obs)
                shm_reward[idx,step] = np.array(reward)
                shm_over[idx,step] = np.array(over)
                info = list(filter(lambda x: x is not None, info))
                remote.send(info)
            elif cmd == "reset":
                obs = [games[i].reset() for i in range(num)]
                shm_obs[idx] = np.stack(obs)
                remote.send(0)
            elif cmd == "close":
                remote.close()
                for i in shms: i[0].close()
                break
            elif cmd == "set_param":
                for i in games:
                    i.reset_params = data
            elif cmd == "get_stats":
                epoch = data
                remote.send(games[0].env.GetClearCol())
            else:
                raise NotImplementedError
    except:
        print(traceback.format_exc())
        raise
    finally:
        remote.close()
        for i in shms: i[0].close()

class Worker:
    """Creates a new worker and runs it in a separate process."""
    def __init__(self, name, shms, idx, seed):
        self.child, parent = Pipe()
        self.process = Process(
                target = worker_process,
                args = (parent, name, shms, idx, seed))
        self.process.start()
