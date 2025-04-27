import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class LlamaConfig:
    hidden_dim: int          # 隐藏层维度
    intermediate_dim: int    # 中间层维度
    dropout_rate: float      # dropout 比率
    bias: bool = True        # 是否使用偏置


class LlamaFNN(nn.Module):
    def __init__(self, config:LlamaConfig)->None:
        super().__init__()
        