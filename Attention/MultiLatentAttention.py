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
    atten_bias:bool = False     # 注意力机制中Q、K、V计算是否需要设置bias, 默认是不需要设置

    q_head_dim:int = qk_nope_head_dim + qk_rope_head_dim
    

class MulitLatentAttention(nn.Module):
    def __init__(self, config:Config)->None:
        super().__init__()

        self.hidden_dim = config.hidden_dim                                                         # 设置隐藏层维度
        self.head_nums = config.head_nums                                                           # 多头的数量
        self.q_head_dim = config.q_head_dim                                                         # 设置q头的隐藏层维度
        self.qk_nope_head_dim = config.qk_nope_head_dim                                             # q和k不带位置位置编码的隐藏层维度
        self.qk_rope_head_dim = config.qk_rope_head_dim                                             # q和k带位置位置编码的隐藏层维度
        self.v_head_dim = config.v_head_dim                                                         # 设置v头的隐藏层维度
        self.dropout = nn.Dropout(config.dropout_rate)                                              # 设置dropout层

        self.q_lora_rank = config.q_lora_rank                                                       # 设置Q的压缩维度
        self.kv_lora_rank = config.kv_lora_rank                                                     # 设置V的压缩维度和不带位置编码的K的压缩维度
        

        # 压缩相关
        self.q_down_proj = nn.Linear(self.hidden_dim, self.q_lora_rank, config.atten_bias)          # Q的压缩矩阵
        self.kv_down_proj = nn.Linear(self.hidden_dim, self.kv_lora_rank + config.qk_rope_head_dim, config.atten_bias)        # 置V的压缩维度和不带位置编码的K的压缩矩阵

        # 升维相关
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.head_nums * self.q_head_dim, config.atten_bias)                     # q的升维矩阵
        # self.q_head_dim - config.qk_rope_head_dim = kv_nope_dim
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.head_nums * ((self.q_head_dim - config.qk_rope_head_dim) + self.v_head_dim), config.atten_bias)         # v和不带位置编码部分的k的升维矩阵

        # 旋转位置编码相关
        self.rope_embedding = None


        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, config.atten_bias)          # 设置输出映射层

        # RMSNorm层相关
        self.q_down_norm = None
        self.kv_down_norm = None

    def forward(self, x:torch.Tensor, position:torch.Tensor, mask:Optional[torch.Tensor]=None)->torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 1. 压缩V
        q = self.q_down_proj(x)
        # q = self.q_down_norm(q)

        # 2. 升维V
        q = self.q_up_proj(q)
        q = q.view(batch_size, seq_len, self.q_head_dim).transpose(1, 2)

        # 3. 分裂q为两部分, 一部分做位置编码, 另一部分不做旋转位置编码
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)     

        # 4. 压缩不带位置编码k和v
        c_kv = self.kv_down_proj(x)
        
        # 5. 先将用于位置编码的k部分和用于压缩v和不带位置编码的k部分 分裂出来
        # 注意: 压缩v和不带位置编码的k部分都放在c_kv中, 现在c_kv会去中升维度, 升维后会将其分成不带位置编码k_nope和v
        # 你可能会为k带位置编码的部分, 按照MLA的图是直接使用x的一部分来做位置编码的, 为什么这里先要将x做一次压缩后，再分裂出一部分来做k的位置编码
        c_kv, k_rope = torch.split(c_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)





        


