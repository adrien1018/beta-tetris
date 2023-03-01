import hashlib, traceback
import numpy as np, torch
from multiprocessing import shared_memory
from torch.multiprocessing import Process, Pipe

import tetris

kTensorDim = tetris.Tetris.StateShape()
kH, kW = kTensorDim[1:]

def worker_process(remote, name: str, shms: list, idx: slice, seed: int):
    shms = [(shared_memory.SharedMemory(name), shape, typ) for name, shape, typ in shms]
    shm_obs, shm_reward, shm_over = [
            np.ndarray(shape, dtype = typ, buffer = shm.buf) for shm, shape, typ in shms]
    # create game environments
    num = idx.stop - idx.start
    manager = tetris.TrainingManager()
    # wait for instructions from the connection and execute them
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                step, actions, epoch = data
                obs, reward, over, info = manager.Step(actions)
                shm_obs[idx] = obs
                shm_reward[idx,step] = reward
                shm_over[idx,step] = over
                remote.send(info)
            elif cmd == "reset":
                seed = int.from_bytes(hashlib.sha256(int.to_bytes(seed, 8, 'little')).digest()[:8], 'little')
                shm_obs[idx] = manager.Init(num, seed)
                remote.send(0)
            elif cmd == "close":
                remote.close()
                for i in shms: i[0].close()
                break
            elif cmd == "set_param":
                manager.SetResetParams(*data)
            elif cmd == "get_stats":
                epoch = data
                remote.send(tetris.GetClearCol())
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
