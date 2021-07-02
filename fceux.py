#!/usr/bin/env python3

import sys, random, time, os.path, socketserver, argparse, math
import numpy as np, torch
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
search_enable = False
first_gain = 0.0

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

    def stepNoSearchAndSend(self, game):
        game.InputPlacement(*GetStrat(model, game), False)
        seq = game.GetMicroadjSequence()
        self.request.send(self.gen_seq(seq))
        game.InputPlacement(*GetStrat(model, game), False)
        seq = game.GetPlannedSequence()
        self.request.send(self.gen_seq(seq))

    def stepAndSend(self, game):
        def searchCallback(state, place_stage, return_value):
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
        if search_enable and game.GetLines() < search_limit:
            w = game.Search(searchCallback, first_gain)
            if w is None:
                self.stepNoSearchAndSend(game)
            else:
                self.request.send(self.gen_seq(w[0]))
                self.request.send(self.gen_seq(w[1]))
        else:
            self.stepNoSearchAndSend(game)

    def handle(self):
        print('connected')
        game = tetris.Tetris(159263)
        while True:
            try:
                data = self.read_until(1)
                if data[0] == 0xff:
                    cur, nxt, _ = self.read_until(3)
                    # adjustable: hz_dev, hz_dev, microadj_delay, drought_mode, start_level, game_over_penalty
                    st = {'hz_avg': hz_avg, 'hz_dev': hz_dev, 'microadj_delay': microadj_delay,
                          'drought_mode': drought_mode, 'start_level': start_level,
                          'game_over_penalty': penalty}
                    game.ResetGame(**st)
                    print()
                    print()
                    print('Current game:')
                    print('Start level: {}, drought mode: {}'.format(start_level, st['drought_mode']))
                    print('Game over penalty:', st['game_over_penalty'])
                    print('Tapping speed:', 'NormalDistribution({}, {})'.format(st['hz_avg'], st['hz_dev']) if st['hz_dev'] > 0 else 'constant {}'.format(st['hz_avg']), 'Hz')
                    print('Microadjustment delay:', st['microadj_delay'], 'frames', flush = True)
                    game.SetNowPiece(cur)
                    game.SetNextPiece(nxt)
                    self.stepAndSend(game)
                elif data[0] == 0xfd:
                    r, x, y, nxt = self.read_until(4)
                    game.SetPreviousPlacement(r, x, y)
                    game.SetNextPiece(nxt)
                    self.stepAndSend(game)
            except ConnectionResetError:
                self.request.close()
                break
            except ValueError:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--hz-avg', type = float)
    parser.add_argument('--hz-dev', type = float)
    parser.add_argument('--microadj-delay', type = int)
    parser.add_argument('--start-level', type = int)
    parser.add_argument('--game-over-penalty', type = float)
    parser.add_argument('--drought-mode', action = 'store_true')
    parser.add_argument('--first-gain', type = float)
    args = parser.parse_args()
    print(args)
    if args.hz_avg is not None: hz_avg = args.hz_avg
    if args.hz_dev is not None: hz_dev = args.hz_dev
    if args.microadj_delay is not None: microadj_delay = args.microadj_delay
    if args.start_level is not None: start_level = args.start_level
    if args.game_over_penalty is not None: penalty = args.game_over_penalty
    if args.first_gain is not None:
        first_gain = args.first_gain
        search_enable = first_gain >= 0
    drought_mode = args.drought_mode
    
    with torch.no_grad():
        c = Configs()
        model = Model(c.channels, c.blocks).to(device)
        model_path = args.model
        if model_path[-3:] == 'pkl': model.load_state_dict(torch.load(model_path)[0].state_dict())
        else: model.load_state_dict(torch.load(model_path))
        model.eval()
    # load GPU first to reduce lag
    GetStrat(model, tetris.Tetris())

    print('Ready')
    HOST, PORT = 'localhost', 3456
    with socketserver.TCPServer((HOST, PORT), GameConn) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
