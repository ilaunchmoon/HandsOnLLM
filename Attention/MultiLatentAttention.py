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



# RMSNorm模块
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-6)->None:
        self.weigt = nn.parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forwrad(self, x:torch.Tensor)->torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weigt
    

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
        将 x 的后一半和前一半进行“旋转”操作： [-x2, x1]
        假设 x.shape = [..., dim]，那么 x 被切为 [..., dim//2] + [..., dim//2]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """
        将旋转位置编码应用到 query 和 key 上。
        q, k: [batch, heads, seq_len, head_dim]
        cos, sin: [seq_len, head_dim]
        position_ids: [batch, seq_len] 位置索引张量
        """
        # 根据 position_ids 提取实际的 cos/sin，维度扩展用于广播：变成 [batch, 1, seq_len, head_dim]
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)

        # 对 q 进行维度重排，使其变成可以旋转的格式
        b, h, s, d = q.shape
        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        # 对 k 做同样处理
        b, h, s, d = k.shape
        k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        # 计算 RoPE 融合：q' = q·cos + rotate_half(q)·sin
        q_embed = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)

        return q_embed, k_embed

        
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
        self.rope_embedding = RotaryEmbedding(config.qk_rope_head_dim, base=config.rope_theta)      # 仅仅放回旋转矩阵


        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, config.atten_bias)          # 设置输出映射层

        # RMSNorm层相关
        self.q_down_norm = RMSNorm(config.q_lora_rank)
        self.kv_down_norm = RMSNorm(config.kv_lora_rank)


    def forward(self, x:torch.Tensor, position:torch.Tensor, mask:Optional[torch.Tensor]=None)->torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 1. 压缩V
        q = self.q_down_proj(x)
        q = self.q_down_norm(q)

        # 2. 升维V
        q = self.q_up_proj(q)
        q = q.view(batch_size, seq_len, self.q_head_dim).transpose(1, 2)

        # 3. 分裂q为两部分, 一部分做位置编码, 另一部分不做旋转位置编码
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)     

        # 4. 压缩不带位置编码k和v
        c_kv = self.kv_down_proj(x)
        
        # 5. 先将用于位置编码的k部分和用于压缩v和不带位置编码的k部分 分裂出来
        # 注意: 压缩v和不带位置编码的k部分都放在c_kv中, 现在c_kv会去中升维度, 升维后会将其分成不带位置编码k_nope和v

        # 你可能会为k带位置编码的部分, 按照MLA的图是直接使用x的一部分来做位置编码的, 为什么这里先要将x做一次压缩后，再分裂出一部分来做k的位置编码？
        # 其实我一直都没想明白, 但是官方也是这么实现的
        c_kv, k_rope = torch.split(c_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_down_norm(self.kv_lora_rank)     # RMSNorm

        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)


        # 6. 对c_kv进行升维操作, 为分裂v部分和不带位置编码的k_nope部分做准备
        kv = self.kv_up_proj(c_kv)


        # 7. 升维度后, 对形状进行重塑: ()
        kv = kv.view(batch_size, seq_len, self.head_nums, self.qk_nope_head_dim + self.v_head_dim).transpose(1,2)

        # 8. 分裂出v和不带位置编码k_nope
        k_nope, v_state = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # 9. 对k_rope 和 q_rope进行位置编码
        cos, sin = self.rope_embedding(v_state, seq_len=v_state.shape[-2])
        q_pos_embeding, k_pos_embeding = RotaryEmbedding.apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position)

        # 10. 对q_nope和q_rope部分进行concat
        # 合并之前, 先将q_rope从(batch_size, seq_len, qk_rope_head_dim) 扩展为(batch_size, head_nums, seq_len, qk_rope_head_dim)
        q_rope = q_rope.expand(-1, self.head_nums, -1, -1)
        q_state = torch.concat([q_nope, q_rope], dim=-1)
        
        # 11. 对k_nope和k_rope部分进行concat
        k_state = torch.concat([k_nope, k_rope], dim=-1)


        # 12. 现在q_state、k_state、v_state都为[batch_size, head_nums, seq_len, head_dim] 其中head_dim = v_head_dim
        # 那么可以进行注意力得分计算了, 和之前的多头注意力机制计算没有任何区别
        att_weight = torch.matmul(q_state, k_state.transpose(-2, -1)) / math.sqrt(self.v_head_dim)

        if mask is not None:
            att_weight = att_weight.masked_fill(mask==0, float('-inf'))
        
        att_weight = F.softmax(att_weight, dim=-1)

        att_weight = self.dropout(att_weight)

        att = torch.matmul(att_weight, v_state)

        att = att.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(att)
    



    





        


