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
    multiple_of: int = 256                  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
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
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))      # 1 / 10000.0 ^ (2i / dim), i为当前位置的词嵌入向量的维度
        t = torch.arange(end, device=freqs.device)  # type: ignore                          # 位置参数
        freqs = torch.outer(t, freqs).float()  # type: ignore                               # 位置参数和预计算的频率的外积
        # 生成复数向量: (self.max_seq_len, self.dim // 2)
        # 其中torch.ones_like(freqs)代表模长, 这里取了freqs形状一致的全1矩阵作为模长
        # freqs代表旋转角度
        # torch.polar()会依据模长和旋转角度生成一个复数, 由于模长为1, 则代表单位复数
        # self.freqs_cis中每一元素为: e^(i * 旋转角度) = cos(旋转角度) + isin(旋转角度)
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
        # 生成形状列表
        # 如果i不是1或者ndim-1, 则当前i索引处就赋值为赋值为1, 其余赋值为x.size(i)
        # 如 x:[2, 2, 3, 4], 则shape = [1, 2, 1, 4]
        # 本质就是想将第1个维度和倒数第二个维度设置为1, 以便满足广播机制的条件
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

        # 将xq和xk转为复数形式
        # 使用torch.view_as_complex()一定要确保最后一个维度的长度为2, 因为转成复数形式是需要实部和虚部, 所以最后一个维度需要2个数配对为一个复数
        # 转变为复数后, 最后一个维度就没有
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        # 获取旋转位置战略的形状
        freqs_cis = RotaryEmbedding.reshape_for_broadcast(freqs_cis, xq_)

        # 应用旋转矩阵, 并且将倒数第2个维度压缩
        # 因为从复数形式转为实数形式, 最后一个维度会重新扩展开, 并且这个维度的长度为2
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
        self.n_heads = args.n_heads                                                         # 注意力头数
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads      # kv注意力头数
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
    


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        初始化前馈神经网络模块。

        参数:
            dim (int): 输入维度。
            hidden_dim (int): 前馈层的隐藏维度。
            multiple_of (int): 确保隐藏维度是该值的倍数。
            ffn_dim_multiplier (float, optional): 隐藏维度的自定义乘数，默认为None。

        属性:
            w1 (nn.Linear): 第一层的线性变换。
            w2 (nn.Linear): 第二层的线性变换。
            w3 (nn.Linear): 第三层的线性变换。
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # 自定义维度因子乘数
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: V2ModelArgs):
        """
        初始化Transformer块

        参数:
            layer_id (int): 层的标识符
            args (ModelArgs): 模型配置参数

        属性:
            n_heads (int): 注意力头的数量
            dim (int): 模型的维度大小
            head_dim (int): 每个注意力头的维度大小
            attention (Attention): 注意力模块
            feed_forward (FeedForward): 前馈神经网络模块
            layer_id (int): 层的标识符
            attention_norm (RMSNorm): 注意力输出的层归一化
            ffn_norm (RMSNorm): 前馈输出的层归一化
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        通过TransformerBlock执行前向传递。

        参数:
            x (torch.Tensor): 输入张量
            start_pos (int): 注意力缓存的起始位置
            freqs_cis (torch.Tensor): 预计算的余弦和正弦频率
            mask (torch.Tensor, optional): 注意力的掩码张量, 默认为None

        返回:
            torch.Tensor: 应用注意力和前馈层后的输出张量
        """
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: V2ModelArgs):
        """
        初始化Transformer模型

        参数:
            params (ModelArgs): 模型配置参数

        属性:
            params (ModelArgs): 模型配置参数
            vocab_size (int): 词汇表大小
            n_layers (int): 模型中的层数
            tok_embeddings (nn.Embedding): 词元嵌入
            layers (torch.nn.ModuleList): Transformer块的列表
            norm (RMSNorm): 模型输出的层归一化
            output (nn.Linear): 最终输出的线性层
            freqs_cis (torch.Tensor): 预计算的余弦和正弦频率
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = RotaryEmbedding.precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        通过Transformer模型执行前向传递

        参数:
            tokens (torch.Tensor): 输入词元索引
            start_pos (int): 注意力缓存的起始位置

        返回:
            torch.Tensor: 应用Transformer模型后的输出逻辑
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output