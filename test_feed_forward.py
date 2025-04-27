import torch
import unittest
from FeedForward.gpt2_ffn import FeedForward, FeedForwardConfig

class TestFeedForward(unittest.TestCase):
    def setUp(self):
        # 设置基本配置
        self.config = FeedForwardConfig(
            hidden_dim=512,        # 隐藏层维度
            intermediate_dim=2048,  # 中间层维度（4倍）
            dropout_rate=0.1,      # dropout 比率
            bias=True              # 使用偏置
        )
        
        # 初始化模型
        self.model = FeedForward(self.config)
        
        # 测试数据维度
        self.batch_size = 4
        self.seq_len = 16

    def test_forward_pass(self):
        # 测试基本的前向传播
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        output = self.model(x)
        
        # 检查输出维度
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_dim))

    def test_intermediate_dimension(self):
        # 测试中间层维度
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        
        # 获取第一层的输出
        intermediate = self.model.c_fc(x)
        
        # 检查中间层维度
        self.assertEqual(intermediate.shape, (self.batch_size, self.seq_len, self.config.intermediate_dim))

    def test_dropout(self):
        # 测试 dropout 功能
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        
        # 训练模式
        self.model.train()
        output_train = self.model(x)
        
        # 评估模式
        self.model.eval()
        output_eval = self.model(x)
        
        # 检查 dropout 是否在训练模式下生效
        self.assertFalse(torch.allclose(output_train, output_eval))

    def test_gelu_activation(self):
        # 测试 GELU 激活函数
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        
        # 获取第一层的输出（在激活函数之前）
        pre_activation = self.model.c_fc(x)
        
        # 获取激活后的输出
        post_activation = self.model.act(pre_activation)
        
        # 检查激活函数是否改变了输出
        self.assertFalse(torch.allclose(pre_activation, post_activation))

    def test_single_sample(self):
        # 测试单个样本的处理
        x = torch.randn(1, 1, self.config.hidden_dim)
        output = self.model(x)
        
        # 检查输出维度
        self.assertEqual(output.shape, (1, 1, self.config.hidden_dim))

    def test_output_range(self):
        # 测试输出值的范围
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        output = self.model(x)
        
        # 检查输出是否在合理范围内
        self.assertTrue(torch.isfinite(output).all())  # 检查是否有无穷大或NaN值
        
        # 检查输出的统计特性
        mean = output.mean().item()
        std = output.std().item()
        self.assertTrue(-10 < mean < 10)  # 均值应该在合理范围内
        self.assertTrue(0 < std < 10)     # 标准差应该为正且在合理范围内

if __name__ == '__main__':
    unittest.main() 