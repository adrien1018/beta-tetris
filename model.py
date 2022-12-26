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

class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                )
        self.final = nn.ReLU(True)
    def forward(self, x):
        return self.final(self.main(x) + x)

class LinearWithChannels(nn.Module):
    __constants__ = ['channels', 'in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, channels:int, in_features: int, out_features: int, bias: bool = True,
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
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
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
    def __init__(self, ch, blk):
        super().__init__()
        self.start = nn.Sequential(
                nn.Conv2d(kBoardChannel, ch, 5, padding = 2),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                )
        self.res1 = nn.Sequential(*[ConvBlock(ch) for i in range(blk-min(2, blk//2))])
        self.mid = nn.Conv2d(ch + kOtherChannel, ch, 3, padding = 1)
        self.res2 = nn.Sequential(*[ConvBlock(ch) for i in range(min(2, blk//2))])
        self.pi_logits_head = nn.Sequential(
                nn.Conv2d(ch, 28, 1),
                nn.BatchNorm2d(28),
                nn.ReLU(True),
                nn.Flatten(2, -1),
                #LinearWithChannels(28, kH * kW, kH * kW),
                nn.Linear(kH * kW, kH * kW),
                nn.Unflatten(1, (7, 4)),
                )
        self.value = nn.Sequential(
                nn.Conv2d(ch, 1, 1),
                nn.BatchNorm2d(1),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(1 * kH * kW, 256),
                nn.ReLU(True),
                )
        self.value_last = nn.Linear(256, 3)

    @autocast()
    def forward(self, obs: torch.Tensor, return_categorical: bool = True):
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
        pieces = torch.argmax(misc1[:,:7], 1)
        indices = pieces.view(-1, 1, 1, 1).repeat(1, 1, 4, kH * kW)
        x = self.start(q1)
        x = self.res1(x)
        x = self.mid(torch.cat([x, q2], 1))
        x = self.res2(x)
        valid = obs[:,9:13].view(batch, -1)
        pi = self.pi_logits_head(x).gather(1, indices).view(batch, -1)
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
