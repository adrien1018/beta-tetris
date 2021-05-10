import math
import torch, numpy as np
from torch import nn
from torch.distributions import Categorical
from torch.cuda.amp import autocast

from game import kH, kW

# refer to Tetris::GetState for details
kOrd = 13
kInChannel = kOrd + 83
kTargetId = 18
kTargetMultiplier = 5e-6 / 5e-6 # kTargetRewardMultipler_ / GetState multiplier

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
                nn.Conv2d(kInChannel, ch, 3, padding = 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(True),
                )
        self.res = nn.Sequential(*[ConvBlock(ch) for i in range(blk)])
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
        q = torch.zeros((obs.shape[0], kInChannel, kH, kW), dtype = torch.float32, device = obs.device)
        q[:,:kOrd] = obs[:,:kOrd]
        misc = obs[:,kOrd].view(obs.shape[0], -1)[:,:kInChannel-kOrd]
        target_mul = misc[:,kTargetId] * kTargetMultiplier
        q[:,kOrd:] = misc.view(obs.shape[0], -1, 1, 1)
        x = self.start(q)
        x = self.res(x)
        valid = obs[:,9:13].view(obs.shape[0], -1)
        pi = self.pi_logits_head(x)
        value = self.value(x)
        with autocast(enabled = False):
            pi = pi.float()
            pi += obs[:,5:9].view(obs.shape[0], -1) * 2
            pi[valid == 0] = -math.inf
            value = self.value_last(value.float())
            value_tot = value[:,0] + value[:,1] * target_mul
            value = torch.cat((value, value_tot.unsqueeze(-1)), -1)
            if return_categorical: pi = Categorical(logits = pi)
            return pi, value

def obs_to_torch(obs: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(obs, dtype = torch.float32, device = device)
