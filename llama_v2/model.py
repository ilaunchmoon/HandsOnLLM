import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class V2ModelArgs:
    dim: int = 4096                         # 隐藏层维度
    n_layers: int = 32                      # transformer的decoder的层数
    n_heads: int = 32                       # 多头数量
    n_kv_heads: Optional[int] = None        # 注意力头的隐藏层维度
    vocab_size: int = -1                    # 由tokenizer来定义
    norm_eps: float = 1e-5                  # RMSnorm的容差值
    
    max_batch_size: int = 32                # 最大batch
    max_seq_len: int = 2048                 # 最长seq长度



class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6)->None:
        super().__init__()
        self.eps = eps                                      # 防止除0的容差
        self.weight = nn.Parameter(torch.ones(dim))         # 用于可学习的缩放因子γ
    
    def _norm(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)       # x 与 RMS归一化后因子相乘

    def forward(self, x:torch.Tensor)->torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight



class RotaryEmbedding:
    """
    用于处理Transformer模型中旋转位置编码的类
    该类提供了计算和应用旋转位置编码的功能
    """
    
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        """
        预计算给定维度的复指数(cis)频率张量:  1 / 10000.0 ^ (2i / dim)

        此函数使用给定的维度'dim'和结束索引'end'计算复指数频率张量
        参数'theta'用于缩放频率
        返回的张量包含complex64数据类型的复数值

        参数:
            dim (int): 频率张量的维度
            end (int): 预计算频率的结束索引
            theta (float, 可选): 频率计算的缩放因子, 默认为10000.0

        返回:
            torch.Tensor: 预计算的复指数频率张量
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis
    
    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        重塑频率张量以便与另一个张量进行广播

        此函数重塑频率张量，使其具有与目标张量'x'相同的形状，目的是在元素级操作期间广播频率张量

        参数:
            freqs_cis (torch.Tensor): 需要重塑的频率张量
            x (torch.Tensor): 用于广播兼容性的目标张量

        返回:
            torch.Tensor: 重塑后的频率张量

        抛出:
            AssertionError: 如果频率张量与预期形状不匹配
            AssertionError: 如果目标张量'x'没有预期数量的维度
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    
    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用给定频率张量将旋转嵌入应用于输入张量

        此函数使用提供的频率张量'freqs_cis'将旋转嵌入应用于给定的查询'xq'和键'xk'张量
        输入张量被重塑为复数，频率张量被重塑以实现广播兼容性
        结果张量包含旋转嵌入，并作为实张量返回

        参数:
            xq (torch.Tensor): 应用旋转嵌入的查询张量
            xk (torch.Tensor): 应用旋转嵌入的键张量
            freqs_cis (torch.Tensor): 预计算的复指数频率张量

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 包含旋转嵌入的修改后查询张量和键张量的元组
        """
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = RotaryEmbedding.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        为多查询注意力重复键和值张量
        
        参数:
            x (torch.Tensor): 形状为[batch_size, seq_len, n_kv_heads, head_dim]的输入张量
            n_rep (int): 重复每个头部的次数
            
        返回:
            torch.Tensor: 形状为[batch_size, seq_len, n_kv_heads * n_rep, head_dim]的重复张量
        """
        torch.repeat_interleave(x, dim=2, repeats=n_rep)


class Attention(nn.Module):
    """多头注意力模块。"""
    def __init__(self, args:V2ModelArgs)->None:
        """
        初始化注意力模块。

        参数:
            args: 模型配置参数，包含以下属性:
                - dim: 模型的维度
                - n_heads: 注意力头的数量
                - n_kv_heads: 键值头的数量: 如果为None, 则等于n_heads
                - max_batch_size: 最大批次大小
                - max_seq_len: 最大序列长度

        属性:
            n_kv_heads (int): 键和值头的数量
            n_rep (int): 本地头部的重复次数
            head_dim (int): 每个注意力头的维度大小
            wq (nn.Linear): 查询的线性变换
            wk (nn.Linear): 键的线性变换
            wv (nn.Linear): 值的线性变换
            wo (nn.Linear): 输出的线性变换
            cache_k (torch.Tensor): 缓存的键
            cache_v (torch.Tensor): 缓存的值
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # 使用标准线性层进行变换
        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        # 初始化缓存
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        注意力模块的前向传播

        参数:
            x (torch.Tensor): 输入张量
            start_pos (int): 缓存的起始位置
            freqs_cis (torch.Tensor): 预计算的频率张量
            mask (torch.Tensor, optional): 注意力掩码张量

        返回:
            torch.Tensor: 经过注意力处理后的输出张量
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 使用RotaryEmbedding类应用旋转位置编码
        xq, xk = RotaryEmbedding.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 确保缓存设备与输入一致
        device = x.device
        if self.cache_k.device != device:
            self.cache_k = self.cache_k.to(device)
            self.cache_v = self.cache_v.to(device)

        # 更新缓存
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # 获取键和值
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # 如果n_kv_heads < n_heads，则使用RotaryEmbedding.repeat_kv重复k/v头
        keys = RotaryEmbedding.repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_heads, head_dim)
        values = RotaryEmbedding.repeat_kv(values, self.n_rep)  # (bs, seqlen, n_heads, head_dim)

        # 转置维度以便进行注意力计算
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
        
        # 应用softmax并计算注意力输出
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        
        # 重塑输出并应用输出投影
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)