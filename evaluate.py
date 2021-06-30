#!/usr/bin/env python3

import argparse
import numpy as np, torch, sys, random, time, os.path
from torch.distributions import Categorical

import tetris

from game import Game, kH, kW, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hz_avg = 12
hz_dev = 0
microadj_delay = 21
start_level = 18
drought_mode = False
penalty = 0.0

def ResetGame(game):
    game.ResetGame(hz_avg = hz_avg, hz_dev = hz_dev, drought_mode = drought_mode,
                   microadj_delay = microadj_delay, start_level = start_level,
                   game_over_penalty = penalty)

def GetSeed(i):
    return (i * 1242973851)

@torch.no_grad()
def Main(model_path):
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    if model_path is None:
        model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth')
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()

    batch_size = 100
    n = 500
    games = [tetris.Tetris(GetSeed(i)) for i in range(batch_size)]
    for i in games: ResetGame(i)
    started = batch_size
    results = []
    rewards = [0. for i in range(batch_size)]
    is_running = [True for i in range(batch_size)]
    while len(results) < n:
        states = [i.GetState() for i, j in zip(games, is_running) if j]
        states = obs_to_torch(np.stack(states), device)
        pi = model(states, False)[0]
        pi = torch.argmax(pi, 1)
        j = 0
        for i in range(batch_size):
            if not is_running[i]: continue
            action = pi[j].item()
            j += 1
            r, x, y = action // 200, action // 10 % 20, action % 10
            rewards[i] += games[i].InputPlacement(r, x, y)
            if games[i].IsOver():
                results.append((games[i].GetScore(), games[i].GetLines()))
                rewards[i] = 0.
                if started < n:
                    games[i] = tetris.Tetris(GetSeed(started))
                    ResetGame(games[i])
                    started += 1
                else:
                    is_running[i] = False
                if len(results) % 200 == 0: print(len(results), '/', n, 'games started')
    s = list(reversed(sorted([i[0] for i in results])))
    for i in range(len(s) - 1):
        for t in range(2500000, 0, -50000):
            if s[i] >= t and s[i+1] < t: print(t, (i + 1) / n)
    print(np.mean(s), np.std(s))
    s = list(reversed(sorted([i[1] for i in results])))
    for i in range(len(s) - 1):
        for t in range(350, 0, -10):
            if s[i] >= t and s[i+1] < t: print(t, (i + 1) / n)
    print(np.mean(s), np.std(s))

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
