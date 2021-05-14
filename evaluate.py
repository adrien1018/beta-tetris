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
    game.ResetGame(reward_multiplier = 1e-5, hz_avg = 12, hz_dev = 0, microadj_delay = 25, first_tap_max = 0)

@torch.no_grad()
def Main():
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth') if len(sys.argv) <= 1 else sys.argv[1]
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()

    batch_size = 64
    n = 1200
    games = [tetris.Tetris(i * 1242979235) for i in range(batch_size)]
    for i in games: ResetGame(i)
    results = []
    while len(results) < n:
        states = [i.GetState() for i in games]
        states = obs_to_torch(np.stack(states), device)
        pi = model(states, False)[0]
        pi = torch.argmax(pi, 1)
        for i in range(batch_size):
            action = pi[i].item()
            r, x, y = action // 200, action // 10 % 20, action % 10
            games[i].InputPlacement(r, x, y)
            if games[i].IsOver():
                results.append((games[i].GetScore(), games[i].GetLines()))
                ResetGame(games[i])
                if len(results) % 20 == 0: print(len(results))
    print(sorted(results))

if __name__ == "__main__": Main()
