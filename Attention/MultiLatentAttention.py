import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    hidden_dim:int              # 隐藏层维度
    head_nums:int               # 多头的数量
    max_seq_len:int             # 最大序列长度
    rope_theta:float            # 旋转位置编码频率基数, 一般是10000.0
    dropout_rate:float          # 注意力得分后的dropout rate
    q_lora_rank:int             # q的压缩矩阵最后一个维度
    qk_rope_head_dim:int        # q和k带旋转位置编码部分的head_dim, 二者是一致的
    kv_lora_rank:int            # k和v压缩矩阵的最后一个维度, 不带位置编码的k和v都是通过同一个压缩矩阵而来
    v_head_dim:int              # v头隐藏层维度, 其实v_head_vim = hidden_dim // head_nums == head_dim, 就是多头注意力机制中每个头的隐藏层维度
    qk_nope_head_dim:int        # 不带位置编码的q和不带位置编码的k的隐藏层维度
    q_head_dim:int              # 其实就是qk_nope_head_dim + qk_rope_head_dim
    atten_bias:bool = False     # 注意力机制中Q、K、V计算是否需要设置bias, 默认是不需要设置



# RMSNorm模块
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6)->None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

# 源码考虑sin和cos的缓存
# 为了简单起见, 此处删除了考虑sin和cos的缓存, 因此就不考虑max_position_embedding最大位置编码的问题
# 一般情况下如下两个静态函数都是单独作为函数, 而非将放置在旋转位置编码的类中
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()  # 初始化 nn.Module 父类

        self.dim = dim  # 头部维度，例如 64
        self.base = base  # RoPE 基数，控制频率递减速率

        # 构造频率倒数：每两个维度共享一个频率（只针对 dim 的前半部分）
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch_size, num_heads, seq_len, head_dim]
        if seq_len is None:
            seq_len = x.size(-2)

        # t: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        # freqs: [seq_len, dim // 2]
        freqs = torch.outer(t, self.inv_freq)

        # emb: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 返回旋转编码的 cos 和 sin（用于与 q/k 融合）
        return (
            emb.cos().to(dtype=x.dtype),  # [seq_len, dim]
            emb.sin().to(dtype=x.dtype),  # [seq_len, dim]
        )

    @staticmethod
    def rotate_half(x):
        """
        将 x 的后一半和前一半进行"旋转"操作： [-x2, x1]
        假设 x.shape = [..., dim]，那么 x 被切为 [..., dim//2] + [..., dim//2]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        """
        将旋转位置编码应用到 query 和 key 上。
        q, k: [batch_size, heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]
        position_ids: [batch, seq_len] 位置索引张量
        """
        # 根据 position_ids 提取实际的 cos/sin
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]

        q_embed = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)

        return q_embed, k_embed

        
class MulitLatentAttention(nn.Module):
    def __init__(self, config:Config)->None:
        super().__init__()

        self.hidden_dim = config.hidden_dim                                                         
        self.head_nums = config.head_nums                                                           
        self.q_head_dim = config.q_head_dim                                                         
        self.qk_nope_head_dim = config.qk_nope_head_dim                                             
        self.qk_rope_head_dim = config.qk_rope_head_dim                                             
        self.v_head_dim = config.v_head_dim                                                         
        self.dropout = nn.Dropout(config.dropout_rate)                                              

        self.q_lora_rank = config.q_lora_rank                                                       
        self.kv_lora_rank = config.kv_lora_rank                                                     

        # 压缩相关
        self.q_down_proj = nn.Linear(self.hidden_dim, self.q_lora_rank, bias=config.atten_bias)          
        self.kv_down_proj = nn.Linear(self.hidden_dim, self.kv_lora_rank + config.qk_rope_head_dim, bias=config.atten_bias)        

        # 升维相关
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.head_nums * self.q_head_dim, bias=config.atten_bias)                     
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.head_nums * ((self.q_head_dim - config.qk_rope_head_dim) + self.v_head_dim), bias=config.atten_bias)         

        # 旋转位置编码相关
        self.rope_embedding = RotaryEmbedding(config.qk_rope_head_dim, base=config.rope_theta)      

        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.atten_bias)          

        # RMSNorm层相关
        self.q_down_norm = RMSNorm(config.q_lora_rank)
        self.kv_down_norm = RMSNorm(config.kv_lora_rank)

    def forward(self, x:torch.Tensor, position:torch.Tensor, mask:Optional[torch.Tensor]=None)->torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 1. 压缩Q
        q = self.q_down_proj(x)
        q = self.q_down_norm(q)

        # 2. 升维Q
        q = self.q_up_proj(q)
        q = q.view(batch_size, seq_len, self.head_nums, self.q_head_dim).transpose(1, 2)

        # 3. 分裂q为两部分, 一部分做位置编码, 另一部分不做旋转位置编码
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)     

        # 4. 压缩不带位置编码k和v
        c_kv = self.kv_down_proj(x)
        
        # 5. 先将用于位置编码的k部分和用于压缩v和不带位置编码的k部分分裂出来
        c_kv, k_rope = torch.split(c_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_down_norm(c_kv)     # RMSNorm

        # 6. 对c_kv进行升维操作, 为分裂v部分和不带位置编码的k_nope部分做准备
        kv = self.kv_up_proj(c_kv)
        kv = kv.view(batch_size, seq_len, self.head_nums, -1).transpose(1, 2)

        # 7. 分裂出v和不带位置编码k_nope
        k_nope, v_state = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # 8. 准备k_rope和q_rope的维度，使其适合位置编码
        k_rope = k_rope.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
        q_rope = q_rope.contiguous()  # [batch_size, head_nums, seq_len, head_dim]

        # 9. 对k_rope和q_rope进行位置编码
        cos, sin = self.rope_embedding(k_rope, seq_len=seq_len)
        q_rope, k_rope = RotaryEmbedding.apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position)

        # 10. 扩展rope部分到所有头
        k_rope = k_rope.expand(-1, self.head_nums, -1, -1)

        # 11. 合并nope和rope部分
        q_state = torch.cat([q_nope, q_rope], dim=-1)
        k_state = torch.cat([k_nope, k_rope], dim=-1)

        # 12. 计算注意力得分
        att_weight = torch.matmul(q_state, k_state.transpose(-2, -1)) / math.sqrt(self.v_head_dim)

        if mask is not None:
            # 扩展 mask 的维度以匹配 att_weight: [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)
            att_weight = att_weight.masked_fill(mask==0, float('-inf'))
        
        att_weight = F.softmax(att_weight, dim=-1)
        att_weight = self.dropout(att_weight)

        # 打印调试信息
        #print(f"att_weight shape: {att_weight.shape}")
        #print(f"v_state shape: {v_state.shape}")
        
        # [batch_size, head_nums, seq_len, seq_len] x [batch_size, head_nums, seq_len, v_head_dim]
        att = torch.matmul(att_weight, v_state)
        #print(f"att shape after matmul: {att.shape}")
        
        # 重新排列维度
        att = att.transpose(1, 2)  # [batch_size, seq_len, head_nums, v_head_dim]
        #print(f"att shape after transpose: {att.shape}")
        
        # 合并最后两个维度
        att = att.reshape(batch_size, seq_len, self.head_nums * self.v_head_dim)  # [batch_size, seq_len, hidden_dim]
        #print(f"att shape after reshape: {att.shape}")
        
        # 确保输出维度正确
        assert att.size(-1) == self.hidden_dim, f"Expected last dimension to be {self.hidden_dim}, but got {att.size(-1)}"
        return self.out_proj(att)
    



    





        


