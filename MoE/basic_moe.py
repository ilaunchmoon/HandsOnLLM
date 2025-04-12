import torch
import torch.nn as nn 
from torch.nn import functional as F


# 最简易版的MoE
class BasicExpert(nn.Module):
    def __init__(self, feat_in:int, hidden_dim:int, feat_out:int)->None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_out)
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.net(x)
    
class BasicMoe(nn.Module):
    def __init__(self, feat_in:int, hidden_dim:int, feat_out:int, num_expert:int)->None:
        super().__init__()
        # gate用于与各个expert的输出相乘, 归一化后作为各个expert的权重: gate可以是一个非常复杂的MLP或Attention模块
        self.gate = nn.Linear(feat_in, num_expert)      
        # 专家组模块
        self.expert_net = nn.ModuleList([
            BasicExpert(feat_in, hidden_dim, feat_out) for _ in range(num_expert)
        ])

    def  forward(self, x:torch.Tensor)->torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 计算门控权重
        gate = self.gate(x)  # [batch_size, seq_len, num_expert]
        expert_weight = F.softmax(gate, dim=-1)     # 归一化后, 可用作各个专家输出的权重
        
        # 获取每个专家的输出
        expert_outputs = []
        for expert in self.expert_net:
            expert_out = expert(x)  # [batch_size, seq_len, feat_out]
            expert_outputs.append(expert_out)
        
        # 将所有专家的输出堆叠在一起
        expert_outs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_expert, seq_len, feat_out]
        
        # 调整专家权重的维度以匹配专家输出
        expert_weight = expert_weight.transpose(1, 2).unsqueeze(-1)  # [batch_size, num_expert, seq_len, 1]
        
        # 计算加权和
        output = (expert_outs * expert_weight).sum(dim=1)  # [batch_size, seq_len, feat_out]
        
        return output






