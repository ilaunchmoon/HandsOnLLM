from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim:int, eps:float = 1e-6)->None:
        super().__init__()
        self.eps = eps                                              # 初始化除数中容差, 防止除数为0
        self.weight = nn.Parameter(torch.ones(hidden_dim))          # 初始化可训练参数, 用于可学习的缩放因子γ
    
    def _norm(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)      # x 与 RMS归一化后因子相乘
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        output = self._norm(x.float()).type_as(x)                   # 对x进行归一化操作
        return output * self.weight                                 # 最后乘以可学习的缩放因子γ
    

class RotatePosEmbedding:
    def __init__(self, dim:int, max_seq_len:int, theta:float=10000.0, device:str="cpu") -> None:
        assert dim % 2 == 0                 # 特征维度需要偶数, 因为需要和选择矩阵配对, 如果为奇数, 否则最后一个维度将无法使用旋转矩阵
        self.dim = dim                      # 特征维度
        self.max_seq_len = max_seq_len      # 支持的最大序列长度
        self.theta = theta                  # 频率调节参数, 默认为10000.0
        self.device = device                # 默认设备
        self.precompute_freqs_cis()        # 调用预先计算的复数形式的旋转向量
    
    def precompute_freqs_cis(self)->torch.Tensor:  # 预先计算好复数形式的旋转向量
        # 1.0 / (10000.0 ^ (2i / dim)): i是每一个token向量的维度索引
        # freqs: (self.dim//2): 旋转角度
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=self.device)[:self.dim//2].float() / self.dim))
        # 位置索引: 即同一批次中的位置索引
        # pos_idx: (self.max_seq_len)
        pos_idx = torch.arange(self.max_seq_len, device=self.device)
        # pos_idx 和 freqs求外积得到 位置索引和旋转角度的乘积 构成一个矩阵
        # (self.max_seq_len, self.dim // 2)
        # 每个元素: pos_idx * freqs 都是位置索引乘旋转角度
        freqs = torch.outer(pos_idx, freqs).float()
        # 生成复数向量: (self.max_seq_len, self.dim // 2)
        # 其中torch.ones_like(freqs)代表模长, 这里取了freqs形状一致的全1矩阵作为模长
        # freqs代表旋转角度
        # torch.polar()会依据模长和旋转角度生成一个复数, 由于模长为1, 则代表单位复数
        # self.freqs_cis中每一元素为: e^(i * 旋转角度) = cos(旋转角度) + isin(旋转角度)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return self.freqs_cis

    
    def reshape_cis_broadcast(self, x:torch.Tensor)->torch.Tensor:
        # 获取x的第1个维度长度, 即序列长度
        seq_len = x.size(1)     
        # 获取x的维度总数
        ndim = x.ndim   
        # 生成形状列表
        # 如果i不是1或者ndim-1, 则当前i索引处就赋值为赋值为1, 其余赋值为x.size(i)
        # 如 x:[2, 2, 3, 4], 则shape = [1, 2, 1, 4]
        # 本质就是想将第1个维度和倒数第二个维度设置为1, 以便满足广播机制的条件
        shape = [1 if i not in (1, ndim-1) else x.size(i) for i in range(ndim)]
        return self.freqs_cis[:seq_len].view(*shape).to(x.device)       # 对前seq_len进行广播
    
    def apply_rotary_embed(self, xq:torch.Tensor, xk:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = xq.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence len {seq_len} out of range max sequence len: {self.max_seq_len}")
        
        # 将xq和xk转为复数形式
        # 使用torch.view_as_complex()一定要确保最后一个维度的长度为2, 因为转成复数形式是需要实部和虚部, 所以最后一个维度需要2个数配对为一个复数
        # 转变为复数后, 最后一个维度就没有
        xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        # 获取旋转张量的形状
        freq_cis = self.reshape_cis_broadcast(xq_complex)

        # 应用旋转矩阵, 并且将倒数第2个维度压缩
        # 因为从复数形式转为实数形式, 最后一个维度会重新扩展开, 并且这个维度的长度为2
        xq_rotate = torch.view_as_real(xq_complex * freq_cis).flatten(-2)
        xk_rotate = torch.view_as_real(xk_complex * freq_cis).flatten(-2)

        return xq_rotate.type_as(xq), xk_rotate.type_as(xk)
    

@dataclass
class V1ModelArgs:
    dim:int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1        # 词汇表大小
    norm_eps:float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class V1Attention(nn.Module):
    def __init__(self, config:V1ModelArgs)->None:
        super().__init__()
        self.dim = config.dim                           # 隐藏层维度
        self.n_heads = config.n_heads                   # 注意力头数
        self.head_dim = config.dim // config.n_heads    # 注意力头的隐藏层维度

        self.w_q = nn.Linear(self.dim, self.dim)        # q权重
        self.w_k = nn.Linear(self.dim, self.dim)        # k权重
        self.w_v = nn.Linear(self.dim, self.dim)        # v权重

        self.w_out = nn.Linear(self.dim, self.dim)      # 输出映射权重

        self.cache_k = torch.zeros(                     # k cache
            config.max_batch_size,
            config.max_seq_len,
            self.n_heads,
            self.head_dim
        )

        self.cache_v = torch.zeros(                     # v cache
            config.max_batch_size,
            config.max_seq_len,
            self.n_heads,
            self.head_dim
        )

        self.pos_emb = RotatePosEmbedding(config.dim, config.max_seq_len)


    def forward(self, x: torch.Tensor, 
                start_pos:int,
                freqs_cis:torch.Tensor,
                mask:Optional[torch.Tensor]
                )->torch.Tensor:
        batch_size, seq_len, _ = x.shape            # 获取词嵌入向量的维度信息: [batch_size, seq_len, hidden_dim]
        x_q = self.w_q(x)                           # 获取Q、K、V矩阵: [batch_size, seq_len, hidden_dim], 要将最后一个维度分成多个头
        x_k = self.w_k(x)   
        x_v = self.w_v(x)

        x_q = x_q.view(batch_size, seq_len, self.n_heads, self.head_dim)        # 转变Q、K、V的张量形状, 便于进行多头注意力计算: [batch_size, seq_len, hidden_dim] ---> [batch_size, seq_len, n_head, head_dim]
        x_k = x_k.view(batch_size, seq_len, self.n_heads, self.head_dim)        
        x_v = x_v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        x_q, x_k = self.pos_emb.apply_rotary_embed(x_q, x_k)                  # 对q、k添加旋转位置编码

        self.cache_k = self.cache_k.to(x_q)                                     # 对k、v进行缓存
        self.cache_v = self.cache_v.to(x_q)
        
        self.cache_k[:batch_size, :start_pos + seq_len] = x_k                   # start_pos + seq_len的缓存部分
        self.cache_v[:batch_size, :start_pos + seq_len] = x_v 

        keys = self.cache_k[:batch_size, :start_pos + seq_len]                  # 取出新添加的seq_len之前, 直接取出之前的kv缓存
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        # 计算注意力得分
        xq = x_q.transpose(1, 2)             # 注意得分计算先将q、k、v转置为: (batch_size, n_head, seq_len, head_dim)即对非批次维度(最后两个维度)进行矩阵乘法运算
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores= torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:                # 掩码操作
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # (batch_size, n_head, seq_len, seq_len)
        # 其实对于分数scores可以使用droput
        output = torch.matmul(scores, values)                   # (batch_size, n_head, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)      # 合并多头注意力机制: [batch_size, seq_len, hidden_dim]
        return self.w_out(output)                               # 输出前的线性层的映射

    

# 解码器中的前馈层
class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int)->None:
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.linear3 = nn.Linear(dim, hidden_dim)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        activate_value = F.silu(self.linear1(x))
        tmp = activate_value * self.linear3(x)
        return self.linear2(tmp)



class TransformerBlock(nn.Module):
    def __init__(self, 
                layer_id:int,               # 注意力机制层id
                arags:V1ModelArgs)->None:
        super().__init__()
        self.n_heads = arags.n_heads        # 注意力头数            
        self.dim = arags.dim                # 隐藏层维度
        self.head_dim = arags.dim // arags.n_heads      # 注意力头的维度
        self.attention = V1Attention(config=arags)      # 注意力机制层
        self.feed_net = FeedForward(self.dim, hidden_dim=4 * self.dim)      # 前馈层先升维为原来的4倍, 再降为原来的维度
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(arags.dim, eps=arags.norm_eps)        # 解码器中注意力层的归一化层, 原始的transformer中归一化是使用的层归一化
        self.ffn_norm = RMSNorm(arags.dim, eps=arags.norm_eps)              # 解码器中前馈层的归一化层, 原始的transformer中归一化是使用的层归一化

    
    def forward(self, 
                x:torch.Tensor, 
                start_pos:int, 
                freqs_cis:torch.Tensor, 
                mask:Optional[torch.Tensor])->torch.Tensor:
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)  # 注意力计算, x + 代表残差连接
        out = h + self.feed_net.forward(self.ffn_norm(h))                                   # 前馈层计算
        return out 


class Transformer(nn.Module):
    def __init__(self, args:V1ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size           # 词汇表大小
        self.n_layers = args.n_layers               # 解码器的层数
        self.token_embedding = nn.Linear(args.vocab_size, args.dim)     # 位置编码
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))        # 创建n层解码器模块
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)                # 归一化
        self.output = nn.Linear(args.dim, args.vocab_size)              # 输出层, 从hidden_dim映射会词汇表的大小
        self.freq_cis_tmp = RotatePosEmbedding(args.dim, args.max_seq_len)
        self.freq_cis = self.freq_cis_tmp.precompute_freqs_cis()        # 调用预计算后的频率值

    
    def forward(self, token:torch.Tensor, start_pos:int)->float:
        _, seq_len = token.shape                                        # 获取token的序列长度
        h = self.token_embedding(token)                                 # 对token进行位置编码
        self.freq_cis = self.freq_cis.to(h.device)                      # 将预先计算的频率转移到和h同一个device上
        freq_cis = self.freq_cis[start_pos:start_pos + seq_len]         # 由于是token by token的过程, 所以后一个token在前面token基础上继续预测的, 则从start_pos + 新输入的seq_len长的序列
        mask = None
        if seq_len > 1:                                                 # 如果新输入token序列seq_len长度大于1, 说明后面的需要使用掩码遮蔽, 则需要构建mask矩阵
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=token.device)     # 
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freq_cis, mask)

        h = self.norm(h)
        out = self.output(h[:, -1, :])                                  # 输出最后一个序列, 因为是token by token的过程
        return out.float()










    
