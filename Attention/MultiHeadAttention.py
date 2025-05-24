import torch
import torch.nn as nn 
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_num:int, dropout_rate:float=0.5)->None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x:torch.Tensor, mask:Optional[torch.Tensor])->torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # (batch_size, seq_len, hidden_dim)
        q:torch.Tensor = self.q_proj(x)
        k:torch.Tensor = self.k_proj(x)
        v:torch.Tensor = self.v_proj(x)

        # (batch_size, seq_len, hidden_dim) --> (batch_size, seq_len, head_num, head_dim)
        # --> (batch_size,head_num, seq_len, head_dim)
        q_state = q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k_state = k.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        v_state = v.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # (batch_size,head_num, seq_len, head_dim) * (batch_size,head_num, head_dim, seq_len)
        # ---> (batch_size,head_num, seq_len, seq_len)
        # 注意这里需要对masked张量进行维度判断, 一定要使得masked维度和attn_score是一致的
        atten_state = torch.matmul(q_state, k_state.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)     # 扩充到4维
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)                   # 扩充到4维
            atten_state = atten_state.masked_fill(
                mask == 0,
                float('-inf')
            )

        # (batch_size,head_num, seq_len, seq_len)
        atten_score = torch.softmax(atten_state, dim=-1)
        atten_score = self.dropout(atten_score)

        # (batch_size,head_num, seq_len, seq_len) * (batch_size,head_num, seq_len, head_dim)    
        # --> (batch_size,head_num, seq_len, head_dim)
        attention = torch.matmul(atten_score, v_state)

        # (batch_size, head_num, seq_len, head_dim)  --> (batch_size, seq_len, hidden_dim)
        # --> (batch_size, seq_len, hidden_dim)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.output_proj(attention)
    



    

        

        

        
        


