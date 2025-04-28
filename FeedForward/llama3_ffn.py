import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class FeedForwardConfig:
    hidden_dim: int          # 隐藏层维度
    intermediate_dim: int    # 中间层维度，通常是 hidden_dim 的 4 倍
    dropout_rate: float      # dropout 比率
    bias: bool = False       # Llama 3 默认不使用偏置

class FeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig)->None:
        super().__init__()
        
        # Llama 3 使用 SwiGLU 激活函数，需要两个并行的投影
        # 门控路径
        self.gate_proj = nn.Linear(
            config.hidden_dim,
            config.intermediate_dim,
            bias=config.bias
        )
        
        # 上投影路径
        self.up_proj = nn.Linear(
            config.hidden_dim,
            config.intermediate_dim,
            bias=config.bias
        )
        
        # 下投影路径 - 将维度压缩回原始维度
        self.down_proj = nn.Linear(
            config.intermediate_dim,
            config.hidden_dim,
            bias=config.bias
        )
        
        # Dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
            
        Returns:
            输出张量，形状为 [batch_size, seq_len, hidden_dim]
        """
        # 计算门控激活值 - 使用 SiLU (Swish) 激活函数
        gate_output = F.silu(self.gate_proj(x))
        
        # 计算上投影值
        up_output = self.up_proj(x)
        
        # 应用门控机制 (SwiGLU) - 元素级乘法
        activation = gate_output * up_output
        
        # 下投影 - 降维回原始维度
        output = self.down_proj(activation)
        
        # 应用 dropout
        output = self.dropout(output)
        
        return output

