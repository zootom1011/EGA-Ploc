from functools import partial
import torch
import torch.nn as nn
from typing import Dict
from models.ETPLoc.utils import build_kwargs_from_config

__all__ = ["build_act"]

class GELUApprox(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.exp(x)))



# register activation function here
REGISTERED_ACT_DICT: Dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(GELUApprox),
    "mish": Mish,
}


def build_act(name: str, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None
