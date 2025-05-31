import torch
import math
import torch.nn as nn 
from typing import Tuple, Optional
from dataclasses import dataclass
from torch.nn import functional as F


@dataclass
class Config:
    hidden_dim:int = 2048                   # 隐藏层维度
    n_heads:int = 32                        # 隐藏层头数
    n_kv_heads: Optional[int] = None        # llama系列都是GQA, n_kv_heads代表有多个query头共享一个kv组, 初始化为None目的是为了实现MHA、GQA、MQA可以互相切换
    vocab_size:int = -1                     # 词汇表大小, 初始化为-1的原因是因为词汇表的大小不是有模型架构决定的, 而是由分词器决定的, 不同的分词器模型对应着不同的词汇表大小
    n_layers:int = 32                       # transformres层数
    multiple_of: int = 256                  # 用于确保 SwiGLU 激活函数的隐藏层大小 是 256 的倍数, 本质是让SwiGLU激活函数的隐藏层大小是2的指数倍数, 因为2的指数倍数能够很好的做好GPU内存对齐 

    # ffn_dim_multiplier为None时, 代表着FFN的隐藏层维度通常是输入维度的4倍
    ffn_dim_multiplier: Optional[float] = None # Transformer中，FFN的隐藏层维度通常是输入维度的4倍, 但在某些模型变体中，这个比例可以有所不同, ffn_dim_multiplier 允许调整这个比例，例如设置为 2/3 可以使前馈网络更小, 所以它允许设置为float类型

    norm_eps:float = 1e-5                   # RMSNorm中, 防止除0的容差值
    rope_theta:float = 500000.0             # 旋转位置编码的参数
    
    max_batch_size:int = 32                 # 模型允许输入的最大批次数
    max_seq_len:int = 2048                  # 模型能够处理的最大序列长度



class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-5)->None:
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))     # RMSNorm可学习参数 γ, 初始化为全1张量
    
    # x:(batch, seq_len, hidden_dim)
    def _norm(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)       # 对x的hidden_dim做均方根归一化
    

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = self._norm(x.float()).as_type(x)
        return out * self.weight
    

class RotaryEmbedding:
    """
    用于处理Transformer模型中旋转位置编码的类
    该类提供了计算和应用旋转位置编码的功能
    """
    
    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
        """
        预计算给定维度的复指数(cis)频率张量:  1 / 500000.0 ^ (2i / dim)

        此函数使用给定的维度'dim'和结束索引'end'计算复指数频率张量
        参数'theta'用于缩放频率
        返回的张量包含complex64数据类型的复数值

        参数:
            dim (int): 频率张量的维度
            end (int): 预计算频率的结束索引
            theta (float, 可选): 频率计算的缩放因子, 默认为500000.0

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
        # self.freqs_cis中每一元素为(欧拉公式): e^(i * 旋转角度) = cos(旋转角度) + isin(旋转角度)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis
    
    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        重塑频率张量以便与另一个张量进行广播

        此函数重塑频率张量，使其具有与目标张量'x'相同的形状，目的是在元素级操作期间广播频率张量
        本质就是将frqs_cis张量的第1个维度和倒数第二个维度的大小都设置为1, 其他维度设置为和张量x的对应维度上的大小一致即可

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
        # 转变为复数后, 最后一个维度就会减少一半
        # 比如 [4, 8, 12, 64]  --> [4, 8, 12, 64 // 2, 2]  --> [4, 8, 12, 32]
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        # 获取旋转位置战略的形状
        freqs_cis = RotaryEmbedding.reshape_for_broadcast(freqs_cis, xq_)

        # 应用旋转矩阵, 并且将倒数第2个维度压缩
        # 因为从复数形式转为实数形式, 最后一个维度会重新扩展开, 并且这个维度的长度为2
        # 如: 原始[2, 10, 8, 64] --> 转为复数形式  [2, 10, 8, 64//2, 2]
        # [2, 10, 8, 32] 消失的一半维度是因为变成了复数形式 ---> 再转为实数 [2, 10, 8, 32, 2]
        # [2, 10, 8, 64]  即最后两个维度会因为复数变为实数而合并
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
        # torch.repeat_interleave(x, dim=2, repeats=n_rep) 可以使用这个来替代
        # 因为这个函数就是为了实现在倒数第2个维度上重复n_rep次这个功能
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )


class FeedForward(nn.Module):
    def __init__(self, in_out_dim:int, hidden_dim:int, multiple_of:int, ffn_dim_multiplier:Optional[float])->None:
        """
        params:
            in_out_dim: 代表FFN的输入和输出维度
            hidden_dim: 代表FFN中间的隐藏维度
            multiple_of: 代表对齐参数, 一般都是256的倍数
            ffn_dim_multiplier: 代表FFN从输入到隐藏的维度的扩大倍数, 原始Transformer中这个倍数是为4, 但是llama3支持任意非整数倍数

            通过 multiple_of 和 ffn_dim_multiplier 来确定从维度 in_out_dim 到 hidden_dim 到底扩大多少倍

            注意llama3源码中, 这里会实施张量的并行计算, 它的做法是如下:

                如果从这一层到下一层是维度上升(即维度扩展操作), 使用张量的列并行计算, 另外如果是从某个线性空间投射到多个子线性空间, 也比较适合使用列并行计算, 如MHA的多头注意计算Q、K、V
                如果从这一层到下一层是维度下降(即维度压缩操作), 使用张量的行并行计算, 另外如果是从多个子线性空间合并结果, 也比较适合使用行并行计算, 如合并多个子线性空间的attn输出层

                总之, 如果从A到B的线性过程, 如果是在特征维度变大了, 即列数变多(升维操作), 那么就使用列并行计算
                     如果从A到B的线性过程, 如果是在批次维度变大了, 即行数变多(压缩操作), 那么就使用行并行计算
                
                为了简化这里不适应张量并行操作
        
        return:
            None
        """
        super().__init__()
        
        hidden_dim = int(2 * hidden_dim / 3)

        # 提高一个从 in_out_dim 到 hidden_dim到底扩大多少倍的自定义操作
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of )

        self.w1 = nn.Linear(in_out_dim, hidden_dim, bias=False)         # 源码中这里是张量列并行计算, 因为这个升维度操作
        self.w2 = nn.Linear(hidden_dim, in_out_dim, bias=False)         # 源码中这里是张量行并行计算, 因为这个压缩维度操作
        self.w3 = nn.Linear(in_out_dim, hidden_dim, bias=False)         # 源码中这里是张量列并行计算, 因为这个升维度操作, w3实际是类似门控的计算, 这个门控机制控制了有多少从self.w1(x)的内容从FFN经过self.w2()输出出去
        

    def forward(self, x:torch.Tensor)->torch.Tensor:
        activate = F.silu(self.w1(x))                                   # 使用激活函数先对输入的x映射后的结果激活
        gate = activate * self.w3(x)                                    # 使用激活后的结果与门控逐元素相乘, 已实现门控机制
        return self.w2(gate)                                            # 最后经过FFN的输出层, 降维输出
    


# llama3的注意力机制本质就是一个GQA(组注意力查询机制)
class Attn(nn.Module):
    def __init__(self, args:Config)->None:
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads              
        self.n_rep = args.n_heads // self.n_kv_heads                                                # GQA中几个kv头, 即它将q分词n_rep组来共享kv
        self.head_dim = args.dim // args.n_heads                                                    # 每个注意力头的维度

        # 使用标准线性层代替并行线性层
        # llama3源码是支持张量并行计算
        # 其中wq、wk、wv都是列并行, wo是行并行
        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        # 缓存张量
        # kv缓存初始为 [max_batch_size, max_seq_lenm kv_heads, head_dim]形状的全0张量
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
    )->torch.Tensor:
        
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 重塑张量，但不再需要考虑本地头数
        # (batch, seq_len, head_num, head_dim)
        xq = xq.view(bsz, seqlen, -1, self.head_dim)
        xk = xk.view(bsz, seqlen, -1, self.head_dim)
        xv = xv.view(bsz, seqlen, -1, self.head_dim)

        # 应用旋转位置编码
        xq, xk = RotaryEmbedding.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 更新缓存
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk         # 将新添加的seq_len长度序列重新添加进缓存, 只更新当前批次大小的数据, 序列维度上，从 start_pos 开始，写入长度为 seqlen 的数据
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv         # 将新添加的seq_len长度序列重新添加进缓存, 只更新当前批次大小的数据, 序列维度上，从 start_pos 开始，写入长度为 seqlen 的数据

        # 获取完整的缓存数据
        keys = self.cache_k[:bsz, : start_pos + seqlen]                 # 从当前批次开始获取完整的KV, 这里面的包含之前已经生成好的, 和当前批次中新添加进来从 start_pos 到 start_pos + seqlen 的部分
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # 重复 K/V 头以匹配 Q 头的数量
        # 即重复n_rep来以实现n_rep个query头共享一组kv头, 因为这n_rep个kv头是直接在这里复制了n_rep次, 就相当于1组kv共享n_rep个query头
        keys = RotaryEmbedding.repeat_kv(keys, self.n_rep)             
        values = RotaryEmbedding.repeat_kv(values, self.n_rep)

        # 转置张量以准备注意力计算
        # (batch, seq_len, head_num, head_dim) -->
        # (batch, head_num, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 计算注意力分数
        # (batch, head_num, seq_len, head_dim) * (batch, head_num, head_dim, seq_len) --> 
        # (batch, head_num, seq_len, seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores + mask          # 注意这里掩饰操作为: 未来待预测的token的位置设置为-inf, 而已经生成的token位置设置为0,
                                            # 从而做到可以关注之前已经生成toekn, 因为任何已生成的token位置上的score不为0, 那么它加一个为0的淹码还是为原来的score
                                            # 但是待预测位置的token的mask值为-inf, 那么任何值加上 -inf, 还是-inf, 从而做到掩码的作用
        
        # 应用 softmax 并计算输出
        # (batch, head_num, seq_len, seq_len) * (batch, head_num, seq_len, head_dim) --> 
        # (batch, head_num, seq_len, head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        
        # 转置并重塑回原始形状
        # (batch, head_num, seq_len, head_dim) -->
        # (batch, seq_len, head_num, head_dim) -->
        # (batch, seq_len, hidden_dim) 
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # 通过输出线性层
        return self.wo(output)
    

class TransformerBlock(nn.Module):
    def __init__(self, layer_id:int, args:Config)->None:
        super().__init__()
        self.n_heads = args.n_heads         # 注意力头数
        self.layer_id = layer_id            # transformer层的id
        self.hidden_dim = args.hidden_dim   # 隐藏层维度
        self.attn = Attn(args=args)         # 注意力计算
        self.ffn = FeedForward(in_out_dim=args.hidden_dim, hidden_dim= 4* args.hidden_dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)        # ffn计算默认还是升高4倍维度

        self.attn_rmsnorm = RMSNorm(args.hidden_dim, eps=args.norm_eps)     # 注意力计算前的RMSNorm
        self.ffn_rmsnorm = RMSNorm(args.hidden_dim, eps=args.norm_eps)      # ffn计算前的RMSNorm

    
    def forward(self, x:torch.Tensor, start_pos:int, freq_cis:torch.Tensor, mask:Optional[torch.Tensor])->torch.Tensor:
        attn_norm = self.attn_rmsnorm(x)        # llama3是Pre-LN, 所以先进行RMSNorm
        attn = self.attn(attn_norm, start_pos, freq_cis, mask)             # 注意力计算
        h = x + attn                                                       # 残差连接
        ffn_norm = self.ffn_rmsnorm(h)                                     # ffn层RMSNorm, llama3是Pre-LN, 所以先进行RMSNorm
        out_ffn = self.ffn(ffn_norm)                                       # ffn层计算
        out = h + out_ffn                                                  # 残差连接
        return out



import torch
import math
from typing import Optional, Tuple
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, params: Config):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # 使用标准的 nn.Embedding 替代 VocabParallelEmbedding
        # llama3中源码使用的是词嵌入并行计算
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # decoder计算
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # RMS-LN层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        
        # 使用标准的 nn.Linear 替代 ColumnParallelLinear
        # llama3中源码使用的是输出层并行计算, 将结果映射到词汇表大小的维度上
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # 预计算位置编码
        self.freqs_cis = RotaryEmbedding.precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int)->torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # 掩码计算
        mask = None
        if seqlen > 1:                                                                      # 如果seq_len大于1, 
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)        # 创建一个全为-inf的张量
            mask = torch.triu(mask, diagonal=1)                                             # 将mask变成一个上三角矩阵, 上三角部分不包含对角线的所有位置还是-inf,代表被遮掩 下三角区域全是0, 代表能够关注
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]              # 将之前已经生成的token位置、当前要生成的token位置，未来要生成的位置都拼接起起来, 如下：
            ).type_as(h)

            """
                [0 0 0 0 | -inf -inf -inf]
                [0 0 0 0 |    0 -inf -inf]
                [0 0 0 0 |    0    0 -inf]
                [0 0 0 0 |    0    0    0]
                ↑            ↑
                历史token   新token
            """

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()             # 将输入转为float类型的张量, 即FP32类型
        return output


