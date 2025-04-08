import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_groups, dropout_rate=0.1):
        """
        初始化组查询注意力模块
        
        参数:
            hidden_dim (int): 隐藏层维度大小
            num_heads (int): 查询的注意力头总数
            num_kv_groups (int): 键值组的数量(必须能够被num_heads整除)
            dropout_rate (float): Dropout概率
        """
        super().__init__()
        
        # 检查头数是否能被组数整除
        if num_heads % num_kv_groups != 0:
            raise ValueError(f"注意力头数({num_heads})必须能被键值组数({num_kv_groups})整除")
        
        self.hidden_dim = hidden_dim                      # 隐藏层维度
        self.num_heads = num_heads                        # 查询的注意力头总数 
        self.num_kv_groups = num_kv_groups                # 键值组的数量
        self.head_dim = hidden_dim // num_heads           # 每个注意力头的维度
        self.heads_per_group = num_heads // num_kv_groups # 每组中的查询头数量
        self.scaling = 1.0 / math.sqrt(self.head_dim)     # 缩放因子，用于注意力分数的缩放
        
        # 为Q、K、V创建线性投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)                         # 查询投影：维持全部头数
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_groups * self.head_dim, bias=False) # 键投影：减少到组数对应的头数
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_groups * self.head_dim, bias=False) # 值投影：减少到组数对应的头数
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)                       # 输出投影：恢复到原始维度
        
        self.dropout = nn.Dropout(dropout_rate)  # 用于注意力权重的dropout
    
    def forward(self, x, mask=None):
        """
        组查询注意力的前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, hidden_dim)
            mask (torch.Tensor, 可选): 注意力掩码，形状为 (batch_size, 1, 1, 1, seq_len)
                                     其中0表示需要掩盖的位置
        
        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape  # 提取输入的批次大小和序列长度
        
        # 将输入投影到查询、键和值空间
        q = self.q_proj(x)  # (batch_size, seq_len, hidden_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, num_kv_groups * head_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, num_kv_groups * head_dim)
        
        # 重塑查询: (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 重塑键和值: (batch_size, seq_len, num_kv_groups, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        
        # 重新排列维度以进行注意力计算
        # (batch_size, seq_len, hidden_dim) --> 将 hidden_dim 分裂为三个维度：num_kv_groups, heads_per_group, head_dim
        # (batch_size, seq_len, num_kv_groups, heads_per_group, head_dim)--> 
        # 然后将第1个维度的seq_len移动到倒数第2个维度, 然后num_kv_groups, heads_per_group分布安置在第1和第2个维度
        # (batch_size, num_kv_groups, heads_per_group, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_kv_groups, self.heads_per_group, self.head_dim)
        q = q.permute(0, 2, 3, 1, 4)
        
        # (batch_size, seq_len, num_kv_groups * head_dim) --> 将num_kv_groups * head_dim分裂成两个维度, 分别将num_kv_groups 和 head_dim 安置在倒数第2和倒数第1的维度上
        # (batch_size, seq_len, num_kv_groups, head_dim)  --> 然后将第1个维度的seq_len和倒数第2个维度交换, 最后再在第2个维度上扩展一个维度
        # 键、值: (batch_size, num_kv_groups, 1, seq_len, head_dim)
        # 注意这里用unsqueeze(2)添加了一个维度，用于后续的广播
        k = k.permute(0, 2, 1, 3).unsqueeze(2)
        v = v.permute(0, 2, 1, 3).unsqueeze(2)
        
        # 计算缩放点积注意力
        # (batch_size, num_kv_groups, heads_per_group, seq_len, seq_len)
        # transpose(-2, -1)用于交换最后两个维度，实现矩阵乘法
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # 应用掩码(如果提供)
        if mask is not None:
            # 掩码预期形状为(batch_size, 1, 1, 1, seq_len)
            # 需要将其广播以匹配attn_weights的形状
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax和dropout
        # softmax应用于最后一个维度(seq_len)，确保每个位置的注意力权重总和为1
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 将注意力权重应用于值
        # (batch_size, num_kv_groups, heads_per_group, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # 将输出重塑回原始维度
        # (batch_size, num_kv_groups, heads_per_group, seq_len, head_dim) --> 将倒数第2个维度交换到第1个维度,
        # (batch_size, seq_len, num_kv_groups, heads_per_group, head_dim) --> 合并第2、3、4个维度为一个维度, 即第三个维度的 hidden_dim
        # (batch_size, seq_len, hidden_dim)
        attn_output = attn_output.permute(0, 3, 1, 2, 4).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # 最终的线性投影
        output = self.out_proj(attn_output)
        
        return output