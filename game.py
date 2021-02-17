import multiprocessing, hashlib, traceback, sys, os, random
import numpy as np
from multiprocessing import connection, shared_memory

import tetris

kTensorDim = tetris.Tetris.StateShape()
kH, kW = kTensorDim[1:]

class Game:
    def __init__(self, seed: int):
        self.args = (0, False)
        self.env = tetris.Tetris(seed)
        self.reset()

    def step(self, action):
        r, x, y = action // 200, action // 10 % 20, action % 10
        reward = self.env.InputPlacement(r, x, y)
        self.reward += reward
        self.length += 0.5

        info = None
        is_over = False
        if self.env.IsOver():
            info = {'reward': self.reward,
                    'score': self.env.GetScore(),
                    'lines': self.env.GetLines(),
                    'length': self.length}
            is_over = True
            self.reset()
        return self.env.GetState(), reward, is_over, info

    def reset(self):
        self.reward = 0.
        self.length = 0.
        self.env.ResetRandom()
        return self.env.GetState()

def worker_process(remote: connection.Connection, shms: list, idx: slice, seed: int):
    #fp = open('logs/%d' % os.getpid(), 'w')
    #os.dup2(fp.fileno(), 1)
    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    shm_obs, shm_reward, shm_over = [
            np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    # create game environments
    num = idx.stop - idx.start
    Seed = lambda x: int.from_bytes(hashlib.sha256(
        int.to_bytes(seed, 4, 'little') + int.to_bytes(x, 4, 'little')).digest(), 'little')
    games = [Game(Seed(i)) for i in range(num)]
    # wait for instructions from the connection and execute them
    rands = [random.random() for i in range(num)]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                step, data, gb = data
                result = []
                for i in range(num):
                    #if rands[i] < 0.001:
                    #    print(gb, 'Game', i, data[i] // 200, data[i] // 10 % 20, data[i] % 10)
                    #    games[i].env.PrintAllState()
                    #    print('', flush = True)
                    result.append(games[i].step(data[i]))
                    if result[-1][3] is not None:
                        rands[i] = random.random()
                obs, reward, over, info = zip(*result)
                shm_obs[idx] = np.stack(obs)
                shm_reward[idx,step] = np.stack(reward)
                shm_over[idx,step] = np.stack(over)
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
    def __init__(self, shms, idx, seed):
        self.child, parent = multiprocessing.Pipe()
        os.environ['LD_PRELOAD'] = '/usr/lib/libasan.so'
        self.process = multiprocessing.Process(
                target = worker_process,
                args = (parent, shms, idx, seed))
        self.process.start()
