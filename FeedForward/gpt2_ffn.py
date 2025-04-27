import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class FeedForwardConfig:
    hidden_dim: int          # 隐藏层维度
    intermediate_dim: int    # 中间层维度，通常是 hidden_dim 的 4 倍
    dropout_rate: float      # dropout 比率
    bias: bool = True        # 是否使用偏置

class FeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig)->None:
        super().__init__()
        
        # 第一层：扩展维度
        self.c_fc = nn.Linear(
            config.hidden_dim,
            config.intermediate_dim,
            bias=config.bias
        )
        
        # 第二层：压缩维度
        self.c_proj = nn.Linear(
            config.intermediate_dim,
            config.hidden_dim,
            bias=config.bias
        )
        
        # Dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # 激活函数：GELU
        self.act = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
            
        Returns:
            输出张量，形状为 [batch_size, seq_len, hidden_dim]
        """
        # 第一层：扩展维度并应用激活函数
        h = self.act(self.c_fc(x))
        
        # 第二层：压缩维度
        h = self.c_proj(h)
        
        # 应用 dropout
        h = self.dropout(h)
        
        return h 