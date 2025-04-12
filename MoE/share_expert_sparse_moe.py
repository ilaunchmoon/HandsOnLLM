import torch
import torch.nn as nn 
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from .basic_moe import BasicExpert
from .sparse_moe import Config, SparseMoe


class SharedExpert(nn.Module):
    def __init__(self, config:Config)->None:
        self.routed_expert_moe = SparseMoe(config)
        self.shared_experts = nn.ModuleList(
            BasicExpert(config.hidden_dim, config.hidden_dim, config.hidden_dim) for _ in range(config.expert_num)
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        


