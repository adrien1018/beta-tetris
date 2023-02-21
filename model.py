import math
import torch, numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
from torch.cuda.amp import autocast

from game import kH, kW

# refer to Tetris::GetState for details
kOrd = 13
kBoardChannel = 1 + 15
kOtherChannel = (kOrd - 1) + (55 - 15)


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(channels, channels, 7, groups=channels, padding=3),
                nn.LayerNorm([channels, kH, kW]),
                nn.Conv2d(channels, channels*4, 1),
                nn.ReLU(True),
                nn.Conv2d(channels*4, channels, 1),
                )

    def forward(self, x):
        return self.main(x) + x


class LinearWithChannels(nn.Module):
    __constants__ = ['channels', 'in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, channels: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearWithChannels, self).__init__()
        self.channels = channels
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((channels, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(channels, 1, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan = init._calculate_correct_fan(self.weight[0], 'fan_in')
        gain = init.calculate_gain('relu')
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (torch.matmul(input.unsqueeze(2), self.weight) + self.bias).squeeze(2)

    def extra_repr(self) -> str:
        return 'channels={}, in_features={}, out_features={}, bias={}'.format(
            self.channels, self.in_features, self.out_features, self.bias is not None
        )


class Model(nn.Module):
    def __init__(self, start_blocks, end_blocks, channels):
        super().__init__()
        total_blocks = start_blocks + end_blocks
        self.start = nn.Sequential(
                nn.Conv2d(kBoardChannel, channels, 5, padding=2),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                *[ConvBlock(channels) for i in range(start_blocks)],
                )
        self.mid_start = nn.Sequential(
                nn.Conv2d(kBoardChannel + kOtherChannel, channels, 5, padding=2),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                )
        self.mid = nn.Sequential(
                *[ConvBlock(channels) for i in range(start_blocks, total_blocks)],
                )
        self.pi_logits_head = nn.Sequential(
                nn.Conv2d(channels, 19, 1),
                nn.LayerNorm([19, kH, kW]),
                nn.Flatten(2, -1),
                nn.ReLU(True),
                LinearWithChannels(19, kH * kW, kH * kW),
                )
        self.value = nn.Sequential(
                nn.Conv2d(channels, 2, 1),
                nn.LayerNorm([2, kH, kW]),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(2 * kH * kW, 128),
                nn.ReLU(True),
                )
        self.value_last = nn.Linear(128, 3)
        # He initialization
        for i in [self.start, self.mid, self.pi_logits_head, self.value]:
            for layer in i:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.value_last.weight, nonlinearity='relu')

    @staticmethod
    def preprocess(obs: torch.Tensor):
        PIECE_MAP = [
            [0,1,2,3],
            [4,5,6,7],
            [8,9,0,0],
            [10,0,0,0],
            [11,12,0,0],
            [13,14,15,16],
            [17,18,0,0],
        ]
        batch = obs.shape[0]
        h, w = obs.shape[2:4]
        # q1, misc1: before intermediate level
        # q2, misc2: after intermediate level
        q1 = torch.zeros((batch, 1 + 15, h, w), dtype = torch.float32, device = obs.device)
        q2 = torch.zeros((batch, (13 - 1) + (55 - 15), h, w), dtype = torch.float32, device = obs.device)
        misc1 = obs[:,13].view(batch, -1)[:,:15]
        misc2 = obs[:,13].view(batch, -1)[:,15:55]
        q1[:,:1] = obs[:,:1]
        q1[:,1:] = misc1.view(batch, -1, 1, 1)
        q2[:,:13-1] = obs[:,1:13]
        q2[:,13-1:] = misc2.view(batch, -1, 1, 1)
        # for gather
        nmap = torch.LongTensor(PIECE_MAP).to(obs.device)
        pieces = torch.argmax(misc1[:,:7], 1)
        indices = torch.arange(batch, device=obs.device).unsqueeze(1), nmap[pieces]
        valid = obs[:,9:13].view(batch, -1)
        return q1, q2, valid, indices

    @autocast()
    def forward(self, obs: torch.Tensor):
        h, w = obs.shape[2:4]
        q1, q2, valid, indices = self.preprocess(obs)
        x = self.start(q1)
        x = self.mid_start(torch.cat([q1, q2], 1)) + x
        x = self.mid(x)
        pi = self.pi_logits_head(x).view(-1, h*w)[indices[0]*19 + indices[1]].view(-1, 4*h*w)
        value = self.value(x)
        with torch.cuda.amp.autocast(enabled=False):
            pi = pi.float()
            pi[valid == 0] = -float('inf')
            value = self.value_last(value.float()).transpose(0, 1)
            value_transform = torch.zeros_like(value)
            value_transform[:2] = value[:2]
            value_transform[2] = torch.nn.functional.softplus(value[2])
            return pi, value_transform


def obs_to_torch(obs: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.float32, device = device)
