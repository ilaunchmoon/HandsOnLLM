import torch
import torch.nn as nn
import math
from typing import Optional

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim:int, num_heads:int, num_kv_groups:int, dropout_rate:float=0.1) -> None:
        """
            hidden_dim: 隐藏层维度
            num_heads: 多头数
            num_kv_groups: kv的组数
            dropout_rate: dropout的概率
        """
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups                      
        self.head_dim = hidden_dim // num_heads
        self.heads_per_group = num_heads // num_kv_groups

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * num_kv_groups)          
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * num_kv_groups)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x:torch.Tensor, mask:Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = x.size()

        # Q: [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_kv_groups, self.heads_per_group, self.head_dim)
        q = q.permute(0, 2, 3, 1, 4)  # [batch_size, G, H_per_G, seq_len, D]

        # K: [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_kv_groups, head_dim]
        k = self.k_proj(x)
        k = k.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [batch_size, G, seq_len, D]
        k = k.unsqueeze(2)  # [batch_size, G, 1, seq_len, D]

        # V: [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_kv_groups, head_dim]
        v = self.v_proj(x)
        v = v.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # [batch_size, G, seq_len, D]
        v = v.unsqueeze(2)  # [batch_size, G, 1, seq_len, D]

        # Attention: [batch_size, G, H_per_G, Q_len, K_len]
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask[:, None, None, None, :]  # [batch_size, 1, 1, 1, seq_len]
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # [batch_size, G, H_per_G, Q_len, D]
        x = torch.matmul(attn, v)
        
        # [batch_size, Q_len, G, H_per_G, D]
        x = x.permute(0, 3, 1, 2, 4)
        
        # [batch_size, Q_len, hidden_dim]
        x = x.reshape(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(x)
