from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim:int, eps:float = 1e-6)->None:
        super().__init__()
        self.eps = eps                                              # 初始化除数中容差, 防止除数为0
        self.weight = nn.Parameter(torch.ones(hidden_dim))          # 初始化可训练参数, 用于可学习的缩放因子γ
    
    def _norm(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)      # x 与 RMS归一化后因子相乘
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        output = self._norm(x.float()).type_as(x)                   # 对x进行归一化操作
        return output * self.weight                                 # 最后乘以可学习的缩放因子γ
    

