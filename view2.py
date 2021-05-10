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

positions = [(2, 19, 1), (1, 17, 0), (2, 19, 4), (2, 19, 7), (0, 17, 7), (1, 14, 0), (0, 17, 3), (1, 17, 5), (1, 15, 8),
                (0, 16, 2), (2, 15, 2), (1, 13, 2), (0, 15, 5), (0, 14, 6), (0, 13, 6), (0, 13, 4), (0, 12, 7), (2, 12, 4),
                (1, 18, 9), (0, 15, 1), (2, 14, 1), (1, 18, 9)]
def InputPosition(game, i):
    if i < len(positions): return positions[i]
    game.PrintState()
    return list(map(int, input().split(' ')))

if __name__ == "__main__":
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth') if len(sys.argv) <= 1 else sys.argv[1]
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()

    game = tetris.Tetris()
    game.ResetGame(reward_multiplier = 1e-5, hz_avg = 12, hz_dev = 0, microadj_delay = 25, first_tap_max = 0)

    pieces = ['J', 'S', 'T', 'L', 'S', 'I', 'S', 'Z', 'J', 'T', 'J', 'T', 'O', 'Z', 'J', 'O', 'J', 'L', 'I', 'T', 'J', 'I', 'O', 'T', 'T', 'Z', 'I', 'S', 'O', 'I', 'Z', 'L']
    game.SetNowPiece(pieces[0])
    game.SetNextPiece(pieces[1])
    if len(positions) == 0: game.PrintState()
    r, x, y = InputPosition(game, 0)
    idx = r*200+x*10+y
    for i in range(len(pieces) - 1):
        game.SetNextPiece(pieces[i + 1])
        if i >= len(positions): game.PrintState()
        pi, val = model(GetTorch(game), False)
        pip = Categorical(logits = pi)
        print(val, pip.probs.max().item(), pip.probs[0,idx].item(), pi[0,idx].item())
        print(game.InputPlacement(r, x, y, False))
        print('-------------------')
        tr, tx, ty = r, x, y

        r, x, y = InputPosition(game, i + 1)
        idx = r*200+x*10+y
        pi, val = model(GetTorch(game), False)
        pip = Categorical(logits = pi)
        print(val, pip.probs.max().item(), pip.probs[0,idx].item(), pi[0,idx].item())
        print(game.InputPlacement(r, x, y, False))
        print('-------------------')
        game.SetPreviousPlacement(tr, tx, ty)
