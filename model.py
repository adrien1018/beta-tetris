import math
import torch, numpy as np
from torch import nn
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

class Model(nn.Module):
    def __init__(self, ch, blk):
        super().__init__()
        self.start = nn.Sequential(
                nn.Conv2d(kBoardChannel, ch, 5, padding = 2),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                )
        self.res1 = nn.Sequential(*[ConvBlock(ch) for i in range(blk-blk//2)])
        self.mid = nn.Conv2d(ch + kOtherChannel, ch, 3, padding = 1)
        self.res2 = nn.Sequential(*[ConvBlock(ch) for i in range(blk//2)])
        self.pi_logits_head = nn.Sequential(
                nn.Conv2d(ch, 8, 1),
                nn.BatchNorm2d(8),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(8 * kH * kW, 4 * kH * kW)
                )
        self.value = nn.Sequential(
                nn.Conv2d(ch, 1, 1),
                nn.BatchNorm2d(1),
                nn.Flatten(),
                nn.ReLU(True),
                nn.Linear(1 * kH * kW, 256),
                nn.ReLU(True),
                )
        self.value_last = nn.Linear(256, 2)

    @autocast()
    def forward(self, obs: torch.Tensor, return_categorical: bool = True):
        batch = obs.shape[0]
        q1 = torch.zeros((batch, kBoardChannel, kH, kW), dtype = torch.float32, device = obs.device)
        q2 = torch.zeros((batch, kOtherChannel, kH, kW), dtype = torch.float32, device = obs.device)
        misc1 = obs[:,kOrd].view(batch, -1)[:,:15]
        misc2 = obs[:,kOrd].view(batch, -1)[:,15:55]
        q1[:,:1] = obs[:,:1]
        q1[:,1:] = misc1.view(batch, -1, 1, 1)
        q2[:,:kOrd-1] = obs[:,1:kOrd]
        q2[:,kOrd-1:] = misc2.view(batch, -1, 1, 1)
        x = self.start(q1)
        x = self.res1(x)
        x = self.mid(torch.cat([x, q2], 1))
        x = self.res2(x)
        valid = obs[:,9:13].view(batch, -1)
        pi = self.pi_logits_head(x)
        value = self.value(x)
        with autocast(enabled = False):
            pi = pi.float()
            pi[valid == 0] = -math.inf
            value = self.value_last(value.float()).reshape(-1, 2).transpose(0, 1)
            if return_categorical: pi = Categorical(logits = pi)
            return pi, value

def obs_to_torch(obs: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.float32, device = device)
