import torch
import unittest
from Attention.MultiLatentAttention import MulitLatentAttention, Config

class TestMultiLatentAttention(unittest.TestCase):
    def setUp(self):
        # 设置基本配置
        self.config = Config(
            hidden_dim=512,          # 隐藏层维度
            head_nums=8,             # 多头数量
            max_seq_len=128,         # 最大序列长度
            rope_theta=10000.0,      # 旋转位置编码频率基数
            dropout_rate=0.1,        # dropout率
            q_lora_rank=64,          # q的压缩矩阵维度
            qk_rope_head_dim=16,     # 带位置编码的q和k的head维度
            kv_lora_rank=64,         # k和v的压缩矩阵维度
            v_head_dim=64,           # v的head维度
            qk_nope_head_dim=48,     # 不带位置编码的q和k的head维度
            q_head_dim=64,           # q的总head维度
            atten_bias=False         # 是否使用偏置
        )
        self.model = MulitLatentAttention(self.config)

    def test_forward_pass(self):
        # 测试基本的前向传播
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.config.hidden_dim)
        position = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        output = self.model(x, position)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_dim))

    def test_attention_mask(self):
        # 测试带掩码的注意力机制
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.config.hidden_dim)
        position = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # 创建上三角掩码
        mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).bool()
        mask = ~mask
        
        output = self.model(x, position, mask)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_dim))

    def test_different_sequence_lengths(self):
        # 测试不同序列长度的输入
        batch_size = 2
        seq_lengths = [5, 15, 20]
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, self.config.hidden_dim)
            position = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            
            output = self.model(x, position)
            
            # 检查输出形状
            self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_dim))

if __name__ == '__main__':
    unittest.main() 