#!/usr/bin/env python3

import sys, random, time, os.path, socketserver
import numpy as np, torch
from torch.distributions import Categorical

import tetris

from game import Game, kH, kW, kTensorDim
from model import Model, ConvBlock, obs_to_torch
from config import Configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GetTorch(game):
    return obs_to_torch(game.GetState(), device).unsqueeze(0)

def GetStrat(model, game):
    with torch.no_grad():
        pi = model(GetTorch(game))[0]
        action = torch.argmax(pi.probs, 1).item()
        return action // 200, action // 10 % 20, action % 10

class GameConn(socketserver.BaseRequestHandler):
    def read_until(self, sz):
        data = b''
        while len(data) < sz:
            n = self.request.recv(sz - len(data))
            if len(n) == 0:
                raise ConnectionResetError()
            data += n
        return data

    @staticmethod
    def frame_to_num(f):
        num = 0
        if f['left']: num |= 1
        if f['right']: num |= 2
        if f['A']: num |= 4
        if f['B']: num |= 8
        return num

    @staticmethod
    def gen_seq(seq):
        if len(seq) == 0: return bytes([0xfe, 1, 0])
        return bytes([0xfe, len(seq)] + [GameConn.frame_to_num(i) for i in seq])

    @staticmethod
    def print_seq(seq):
        ss = ''
        for i in seq:
            s = ''
            for h, l in zip(['left', 'right', 'A', 'B'], 'LRAB'):
                if i[h]: s += l
            if s == '': s = '-'
            ss += s + ' '
        print(ss)

    def handle(self):
        print('connected')
        game = tetris.Tetris(12490)
        while True:
            try:
                data = self.read_until(1)
                if data[0] == 0xff:
                    cur, nxt, level = self.read_until(3)
                    # adjustable: reward_multiplier, hz_dev, hz_dev, microadj_delay, target, start_level
                    game.ResetGame(reward_multiplier = 1e-5, hz_avg = 12, hz_dev = 0,
                                microadj_delay = 25, start_level = level, target = 1000000)
                    game.SetNowPiece(cur)
                    game.SetNextPiece(nxt)
                    print('start cur {} nxt {} level {}'.format(cur, nxt, level))
                    game.InputPlacement(*GetStrat(model, game), False)
                    seq = game.GetMicroadjSequence()
                    self.print_seq(seq)
                    self.request.send(self.gen_seq(seq))
                    game.InputPlacement(*GetStrat(model, game), False)
                    seq = game.GetPlannedSequence()
                    self.print_seq(seq)
                    self.request.send(self.gen_seq(seq))
                elif data[0] == 0xfd:
                    r, x, y, nxt = self.read_until(4)
                    print('pos ({}, {}, {}) nxt {}'.format(r, x, y, nxt))
                    game.SetPreviousPlacement(r, x, y)
                    game.SetNextPiece(nxt)
                    # game.PrintState()
                    game.InputPlacement(*GetStrat(model, game), False)
                    seq = game.GetMicroadjSequence()
                    self.print_seq(seq)
                    self.request.send(self.gen_seq(seq))
                    game.InputPlacement(*GetStrat(model, game), False)
                    seq = game.GetPlannedSequence()
                    self.print_seq(seq)
                    self.request.send(self.gen_seq(seq))
            except ConnectionResetError:
                self.request.close()
                break
            except ValueError:
                pass

if __name__ == "__main__":
    with torch.no_grad():
        c = Configs()
        model = Model(c.channels, c.blocks).to(device)
        model_path = os.path.join(os.path.dirname(sys.argv[0]), 'models/model.pth') if len(sys.argv) <= 1 else sys.argv[1]
        if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
        else: model.load_state_dict(torch.load(model_path))
        model.eval()
    # load GPU first to reduce lag
    print('meow')
    GetStrat(model, tetris.Tetris())

    print('Ready')
    HOST, PORT = 'localhost', 3456
    with socketserver.TCPServer((HOST, PORT), GameConn) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
