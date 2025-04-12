import torch
import torch.nn as nn 
from torch.nn import functional as F
from .basic_moe import BasicExpert
from .sparse_moe import Config, SparseMoe


class SharedExpert(nn.Module):
    def __init__(self, config:Config)->None:
        super().__init__()
        self.routed_expert_moe = SparseMoe(config)
        self.shared_experts = nn.ModuleList([
            BasicExpert(config.hidden_dim, config.hidden_dim, config.hidden_dim) for _ in range(config.shared_expert)
        ])

    def forward(self, x:torch.Tensor)->torch.Tensor:
        batch_size, seq_len, hidden_dim = x.size()

        # 计算共享专家的输出
        shared_expert_outputs = [
            expert(x) for expert in self.shared_experts
        ]

        # 求和得到共享专家的总输出
        shared_expert_out = sum(shared_expert_outputs)
        
        # 计算路由专家的输出
        sparse_moe_out, router_logits = self.routed_expert_moe(x)

        # 组合输出
        output = shared_expert_out + sparse_moe_out

        return output, router_logits



