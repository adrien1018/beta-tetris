#!/usr/bin/env python3

import numpy as np, torch, sys, random, time, os.path
import tetris

from game import Game, kH, kW, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetTorch(game):
    return obs_to_torch(game.env.GetState(), device).unsqueeze(0)

def GetStrat(game):
    pi = model(GetTorch(game))[0]
    act = torch.argmax(pi.probs, 1).item()
    return act

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth') if len(sys.argv) <= 1 else sys.argv[1]
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()
    while True:
        seed = random.randint(0, 2**32-1)
        game = Game(seed)
        while True:
            _, _, x, y = game.step(GetStrat(game))
            if x: break
        if y['lines'] <= 140: continue
        game = Game(seed)
        while True:
            act = GetStrat(game)
            if game.step(act)[2]: break
            game.env.PrintState()
            input()