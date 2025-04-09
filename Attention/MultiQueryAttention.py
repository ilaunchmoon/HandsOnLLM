import torch
import math
import torch.nn as nn 
from torch.nn import functional as F
from typing import Optional


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim:int, head_nums:int, num_kv_group:int=1, dropout_rate:float=0.1)->None:
        super().__init__()
        num_kv_group = 1                                            # 注意由于是MQA，所以只分为一组,
        self.hidden_dim = hidden_dim
        self.head_nums = head_nums
        self.head_dim = hidden_dim // head_nums
        self.num_kv_group = num_kv_group
        self.head_num_per_group = head_nums // num_kv_group
        assert self.num_kv_group == 1                               # 验证组数必须为1
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_group * self.head_dim)      # k和v分为一组self.num_kv_group值为1
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_group * self.head_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    

    def forward(self, x:torch.Tensor, mask:Optional[torch.Tensor]=None)->torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q:torch.Tensor = self.q_proj(x)
        k:torch.Tensor = self.k_proj(x)
        v:torch.Tensor = self.v_proj(x)

        # 重塑q、k、v的形状
        # (batch_size, seq_len, hidden_dim) -->
        # (batch_size, seq_len, num_kv_group, head_num_per_group, head_dim) --> 
        # (batch_size, num_kv_group, head_num_per_group, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_kv_group, self.head_num_per_group, self.head_dim)
        q = q.permute(0, 2, 3, 1, 4)

        # (batch_size, seq_len, num_kv_group * head_dim) -->
        # (batch_size, seq_len, num_kv_group, head_dim) --> 
        # (batch_size, num_kv_group, seq_len, head_dim) -->
        # (batch_size, num_kv_group, 1, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_group, self.head_dim)
        k = k.permute(0, 2, 1, 3).unsqueeze(2)

        # (batch_size, seq_len, num_kv_group * head_dim) -->
        # (batch_size, seq_len, num_kv_group, head_dim) --> 
        # (batch_size, num_kv_group, seq_len, head_dim) -->
        # (batch_size, num_kv_group, 1, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_group, self.head_dim)
        v = v.permute(0, 2, 1, 3).unsqueeze(2)


        # (batch_size, num_kv_group, head_num_per_group, seq_len, seq_len)
        att_weight = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            att_weight = att_weight.masked_fill(
                mask==0.0,
                float("-inf")
            )
        
        # (batch_size, num_kv_group, head_num_per_group, seq_len, seq_len)
        att_weight = F.softmax(att_weight, dim=-1)
        att_weight = self.dropout(att_weight)

        # (batch_size, num_kv_group, head_num_per_group, seq_len, head_dim) -->
        # (batch_size, seq_len, num_kv_group, head_num_per_group, head_dim) -->
        # (batch_size, seq_len, hidden_dim)
        att = torch.matmul(att_weight, v)
        att = att.permute(0, 3, 1, 2, 4).contiguous()
        att = att.view(batch_size, seq_len, -1)

        return self.out_proj(att)


        