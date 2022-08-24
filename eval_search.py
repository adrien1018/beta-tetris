#!/usr/bin/env python3

import argparse
import numpy as np, torch, sys, random, time, os.path, math
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
step_points = 100.
first_gain = 0.0

def ResetGame(game):
    game.ResetGame(hz_avg = hz_avg, hz_dev = hz_dev, drought_mode = drought_mode,
                   microadj_delay = microadj_delay, start_level = start_level,
                   step_points = step_points)

def GetSeed(i):
    return (i * 42973851 + 35)

@torch.no_grad()
def Main(model_path):
    c = Configs()
    model = Model(c.channels, c.blocks).to(device)
    if model_path is None:
        model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth')
    if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
    else: model.load_state_dict(torch.load(model_path))
    model.eval()

    def SearchCallback(state, place_stage, return_value):
        with torch.no_grad():
            states = obs_to_torch(state, device)
            if return_value:
                return model(states, False)[1].tolist()
            else:
                k = 1 if not place_stage and microadj_delay == 61 else 3
                pi = model(states, False)[0]
                ret = torch.topk(pi, k)[1].tolist()
                for i, x in enumerate(ret):
                    while len(ret[i]) > 0:
                        if pi[i, x[-1]] == -math.inf:
                            x.pop()
                        else:
                            break
                return ret

    search_limit = 322
    if start_level == 29 and microadj_delay >= 20: search_limit = 0
    elif drought_mode: search_limit = 222
    elif microadj_delay >= 20: search_limit = 222

    n = 100
    results = []
    for i in range(n):
        # if i % 5 == 0: print(i, file=sys.stderr)
        game = tetris.Tetris(GetSeed(i))
        ResetGame(game)
        while game.GetLines() < search_limit:
            x = game.Search(SearchCallback, first_gain)
            if x is None: break
            # s = torch.argmax(model(obs_to_torch(game.GetState(), device).unsqueeze(0), False)[0], 1).item()
            # game.InputPlacement(s // 200, s // 10 % 20, s % 10)
            # if game.IsOver(): break
        while not game.IsOver():
            s = torch.argmax(model(obs_to_torch(game.GetState(), device).unsqueeze(0), False)[0], 1).item()
            game.InputPlacement(s // 200, s // 10 % 20, s % 10)
        results.append((game.GetScore(), game.GetLines()))
        print(game.GetScore(), file=sys.stderr)

    s = list(reversed(sorted([i[0] for i in results])))
    s = [i[0] for i in results]
    print(s)
    for i in range(len(s) - 1):
        for t in range(2500000, 0, -50000):
            if s[i] >= t and s[i+1] < t: print(t, (i + 1) / n)
    print(np.mean(s), np.std(s), np.std(s) / (n ** 0.5))
    s = list(reversed(sorted([i[1] for i in results])))
    for i in range(len(s) - 1):
        for t in range(350, 0, -10):
            if s[i] >= t and s[i+1] < t: print(t, (i + 1) / n)
    print(np.mean(s), np.std(s), np.std(s) / (n ** 0.5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--hz-avg', type = float)
    parser.add_argument('--hz-dev', type = float)
    parser.add_argument('--microadj-delay', type = int)
    parser.add_argument('--start-level', type = int)
    parser.add_argument('--step-points', type = float)
    parser.add_argument('--drought-mode', action = 'store_true')
    parser.add_argument('--first-gain', type = float)
    args = parser.parse_args()
    print(args)
    if args.hz_avg is not None: hz_avg = args.hz_avg
    if args.hz_dev is not None: hz_dev = args.hz_dev
    if args.microadj_delay is not None: microadj_delay = args.microadj_delay
    if args.start_level is not None: start_level = args.start_level
    if args.step_points is not None: step_points = args.step_points
    if args.first_gain is not None: first_gain = args.first_gain
    drought_mode = args.drought_mode
    Main(args.model)
