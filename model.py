import math
import torch, numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
from torch.distributions import Categorical
from torch.cuda.amp import autocast

from game import kH, kW

# refer to Tetris::GetState for details
kOrd = 13
kBoardChannel = 1 + 15
kOtherChannel = (kOrd - 1) + (55 - 15)


# Global pooling in KataGo
class GlobalPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels < in_channels
        pool_channels = in_channels - out_channels
        self.out_channels = out_channels
        self.before = nn.Sequential(
                nn.BatchNorm2d(pool_channels),
                nn.ReLU(True),
                )
        self.after = nn.Linear(2 * pool_channels, out_channels)

    def forward(self, x):
        mid = self.before(x[:,self.out_channels:])
        pool = torch.cat([mid.mean((2, 3)), mid.amax((2, 3))], 1)
        return self.after(pool).view(-1, self.out_channels, 1, 1) + x[:,:self.out_channels]


class ConvBlock(nn.Module):
    def __init__(self, channels, global_pooling_channels = None):
        super().__init__()
        back_channels = global_pooling_channels or channels
        self.main = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Conv2d(channels, channels, 3, padding = 1),
                *([GlobalPooling(channels, back_channels)] if global_pooling_channels else []),
                nn.BatchNorm2d(back_channels),
                nn.ReLU(True),
                nn.Conv2d(back_channels, channels, 3, padding = 1),
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


PIECE_MAP = [
    [0,1,2,3],
    [4,5,6,7],
    [8,9,0,0],
    [10,0,0,0],
    [11,12,0,0],
    [13,14,15,16],
    [17,18,0,0],
]


def preprocess(obs: torch.Tensor):
    batch = obs.shape[0]
    # q1, misc1: before intermediate level
    # q2, misc2: after intermediate level
    q1 = torch.zeros((batch, kBoardChannel, kH, kW), dtype = torch.float32, device = obs.device)
    q2 = torch.zeros((batch, kOtherChannel, kH, kW), dtype = torch.float32, device = obs.device)
    misc1 = obs[:,kOrd].view(batch, -1)[:,:15]
    misc2 = obs[:,kOrd].view(batch, -1)[:,15:55]
    q1[:,:1] = obs[:,:1]
    q1[:,1:] = misc1.view(batch, -1, 1, 1)
    q2[:,:kOrd-1] = obs[:,1:kOrd]
    q2[:,kOrd-1:] = misc2.view(batch, -1, 1, 1)
    # for gather
    nmap = torch.LongTensor(PIECE_MAP).to(obs.device)
    pieces = torch.argmax(misc1[:,:7], 1)
    indices = torch.arange(batch, device = obs.device).unsqueeze(1), nmap[pieces]
    valid = obs[:,9:13].view(batch, -1)
    return q1, q2, valid, indices


class Model(nn.Module):
    def __init__(self, start_blocks, end_blocks, channels,
                 global_pooling_channels, num_global_pooling):
        super().__init__()
        total_blocks = start_blocks + end_blocks
        if num_global_pooling * total_blocks // (num_global_pooling + 1) < start_blocks:
            global_pooling_positions = {(i * start_blocks - 1) // num_global_pooling for i in range(1, num_global_pooling)}
            global_pooling_positions.add(start_blocks)
        else:
            global_pooling_positions = {(i * total_blocks - 1) // (num_global_pooling + 1) for i in range(1, num_global_pooling + 1)}

        self.start = nn.Sequential(
                nn.Conv2d(kBoardChannel, channels, 5, padding = 2),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                *[ConvBlock(channels, global_pooling_channels if i in global_pooling_positions else None)
                  for i in range(start_blocks)],
                )
        self.mid = nn.Sequential(
                nn.Conv2d(channels + kOtherChannel, channels, 3, padding = 1),
                *[ConvBlock(channels, global_pooling_channels if i in global_pooling_positions else None)
                  for i in range(start_blocks, total_blocks)],
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                )
        self.pi_logits_head = nn.Sequential(
                nn.Conv2d(channels, 38, 1),
                GlobalPooling(38, 19),
                nn.BatchNorm2d(19),
                nn.ReLU(True),
                nn.Flatten(2, -1),
                LinearWithChannels(19, kH * kW, kH * kW),
                )
        self.value = nn.Sequential(
                nn.Conv2d(channels, 1, 1),
                nn.BatchNorm2d(1),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(1 * kH * kW, 256),
                nn.ReLU(True),
                )
        self.value_last = nn.Linear(256, 3)
        # He initialization
        for i in [self.start, self.mid, self.pi_logits_head, self.value]:
            for layer in i:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.value_last.weight, nonlinearity='relu')

    @autocast()
    def forward(self, obs: torch.Tensor, return_categorical: bool = True):
        q1, q2, valid, indices = preprocess(obs)
        x = self.start(q1)
        x = self.mid(torch.cat([x, q2], 1))
        pi = self.pi_logits_head(x)[indices].view(-1, 4 * kH * kW)
        value = self.value(x)
        with autocast(enabled = False):
            pi = pi.float()
            pi[valid == 0] = -math.inf
            value = self.value_last(value.float()).transpose(0, 1)
            value_transform = torch.zeros_like(value)
            value_transform[:2] = value[:2]
            value_transform[2] = F.softplus(value[2])
            if return_categorical: pi = Categorical(logits = pi)
            return pi, value_transform


def obs_to_torch(obs: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.float32, device = device)
