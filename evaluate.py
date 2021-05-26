#!/usr/bin/env python3

import numpy as np, torch, sys, random, time, os.path
from torch.distributions import Categorical

import tetris

from game import Game, kH, kW, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetTorch(game):
    return obs_to_torch(game.GetState(), device).unsqueeze(0)

def GetStrat(game):
    pi = model(GetTorch(game))[0]
    act = torch.argmax(pi.probs, 1).item()
    return act

def ResetGame(game):
    # adjustable: reward_multiplier, hz_dev, hz_dev, microadj_delay, target, start_level
    game.ResetGame(reward_multiplier = 1e-5, hz_avg = 13, hz_dev = 1,
                   microadj_delay = 25, start_level = 18, target = 1100000)

def GetSeed(i):
    return (i * 1242979235)

@torch.no_grad()
def Main():
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth') if len(sys.argv) <= 1 else sys.argv[1]
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()

    batch_size = 100
    n = 1200
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
            rewards[i] += games[i].InputPlacement(r, x, y)[1]
            if games[i].IsOver():
                results.append((games[i].GetScore(), games[i].GetLines()))
                rewards[i] = 0.
                if started < n:
                    games[i] = tetris.Tetris(GetSeed(i))
                    ResetGame(games[i])
                else:
                    is_running[i] = False
                if len(results) % 40 == 0: print(len(results))
    s = list(reversed(sorted([i[0] for i in results])))
    for i in range(len(s) - 1):
        for t in range(1500000, 700000, -50000):
            if s[i] >= t and s[i+1] < t: print(t, (i + 1) / n)
    s = list(reversed(sorted([i[1] for i in results])))
    for i in range(len(s) - 1):
        for t in range(330, 150, -10):
            if s[i] >= t and s[i+1] < t: print(t, (i + 1) / n)

if __name__ == "__main__": Main()
