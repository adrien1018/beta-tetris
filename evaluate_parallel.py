#!/usr/bin/env python3

import argparse
import numpy as np, torch, sys, random, time, os.path, pickle
from torch.distributions import Categorical
from torch.multiprocessing import Process, Pipe

import tetris

from game import Game, kH, kW, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hz_avg = 12
hz_dev = 0
microadj_delay = 21
start_level = 18
drought_mode = False
penalty = -2.0

def ResetGame(game):
    game.ResetGame(hz_avg = hz_avg, hz_dev = hz_dev, drought_mode = drought_mode,
                   microadj_delay = microadj_delay, start_level = start_level,
                   game_over_penalty = penalty)

#def GetSeed(i):
#    return (i * 1242973851 + 1)
def GetSeed(i):
    return (i * 42973851 + 45)

def worker_process(remote, q_size, id):
    games = [tetris.Tetris(GetSeed(i)) for i in range(q_size)]
    rewards = [0. for i in range(q_size)]
    is_running = [True for i in range(q_size)]
    state = 0

    curseed = [0 for _ in range(q_size)]

    while True:
        cmd, data = remote.recv()
        if cmd == 'init':
            assert state == 0
            assert len(data) == q_size
            for i, seed in enumerate(data):
                games[i] = tetris.Tetris(GetSeed(seed))
                ResetGame(games[i])
                curseed[i] = seed
            state = 1
            remote.send(0)
        elif cmd == 'query':
            assert state != 0
            states = [i.GetState() for i, j in zip(games, is_running) if j]
            if len(states) == 0:
                remote.send(None)
            else:
                remote.send(np.stack(states))
        elif cmd == 'step':
            assert state == 1
            j = 0
            results = []
            over = []
            for i in range(q_size):
                if not is_running[i]: continue
                action = data[j]
                j += 1
                r, x, y = action // 200, action // 10 % 20, action % 10
                rewards[i] += games[i].InputPlacement(r, x, y)
                if games[i].IsOver():
                    results.append((games[i].GetScore(), games[i].GetLines()))
                    rewards[i] = 0.
                    over.append(i)
            state = 2
            remote.send(results)
        elif cmd == 'reset':
            assert state == 2
            assert len(over) == len(data)
            for i, seed in zip(over, data):
                if seed is None:
                    is_running[i] = False
                else:
                    games[i] = tetris.Tetris(GetSeed(seed))
                    ResetGame(games[i])
                    curseed[i] = seed
            state = 1
            remote.send(0)
        elif cmd == "close":
            remote.close()
            return

class Worker:
    def __init__(self, q_size, i):
        self.child, parent = Pipe()
        self.process = Process(
                target = worker_process,
                args = (parent, q_size, i))
        self.process.start()

@torch.no_grad()
def Main(model_path):
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    if model_path is None:
        model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth')
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()

    batch_size = 512
    n = 1000
    q_size = 256
    assert batch_size % q_size == 0
    n_workers = batch_size // q_size
    workers = [Worker(q_size, i) for i in range(n_workers)]

    for i in range(n_workers):
        workers[i].child.send(('init', range(i * q_size, (i + 1) * q_size)))
    for i in workers: i.child.recv()
    started = batch_size
    results = []

    st = set()
    #with open('logs/boards30.bin', 'rb') as f:
    #    while True:
    #        item = f.read(25)
    #        if len(item) == 0: break
    #        assert len(item) == 25
    #        st.add(item)
    #of = open('logs/progress/30-4.txt', 'w')

    while len(results) < n:
        for i in workers: i.child.send(('query', None))
        states = [i.child.recv() for i in workers]
        state_lens = [0 if i is None else i.shape[0] for i in states]
        states = [i for i in states if i is not None]
        states = obs_to_torch(np.concatenate(states), device)

        states_num = states[:,0].view(-1, 200)
        states_num = states_num.cpu().numpy().astype(bool)
        states_num = np.packbits(states_num, axis = 1, bitorder = 'little')
        st.update(map(lambda x: x.tobytes(), states_num))
        #print(len(st), file=of, flush=True)

        pi = model(states, False)[0]
        pi = torch.argmax(pi, 1)

        j = 0
        for i in range(n_workers):
            workers[i].child.send(('step', pi[j:j+state_lens[i]].view(-1).cpu().numpy()))
            j += state_lens[i]
        old_started = len(results)
        for i in workers:
            worker_res = i.child.recv()
            results += worker_res
            ids = []
            for _ in worker_res:
                if started < n:
                    ids.append(started)
                    started += 1
                else:
                    ids.append(None)
            i.child.send(('reset', ids))
        for i in workers: i.child.recv()
        if old_started // 50 != len(results) // 50: print(len(results), '/', n, 'games started')

    for i in workers:
        i.child.send(('close', None))
        i.child.close()

    #with open('logs/boards30.bin', 'wb') as f:
    #    for i in sorted(st): f.write(i)
    #of.close()

    s = list(reversed(sorted([i[0] for i in results])))
    print(s)
    mx = s[0] // 50000 * 50000
    for i in range(len(s) - 1):
        for t in range(mx, 0, -50000):
            if s[i] >= t and s[i+1] < t: print(t, (i + 1) / n)
    print(np.mean(s), np.std(s), np.std(s) / (n ** 0.5))
    s = list(reversed(sorted([i[1] for i in results])))
    mx = s[0] // 10 * 10
    for i in range(len(s) - 1):
        for t in range(mx, 0, -10):
            if s[i] >= t and s[i+1] < t: print(t, (i + 1) / n)
    print(np.mean(s), np.std(s), np.std(s) / (n ** 0.5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--hz-avg', type = float)
    parser.add_argument('--hz-dev', type = float)
    parser.add_argument('--microadj-delay', type = int)
    parser.add_argument('--start-level', type = int)
    parser.add_argument('--game-over-penalty', type = float)
    parser.add_argument('--drought-mode', action = 'store_true')
    args = parser.parse_args()
    print(args)
    if args.hz_avg is not None: hz_avg = args.hz_avg
    if args.hz_dev is not None: hz_dev = args.hz_dev
    if args.microadj_delay is not None: microadj_delay = args.microadj_delay
    if args.start_level is not None: start_level = args.start_level
    if args.game_over_penalty is not None: penalty = args.game_over_penalty
    drought_mode = args.drought_mode
    Main(args.model)
