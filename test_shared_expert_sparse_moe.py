import torch
import unittest
from MoE.share_expert_sparse_moe import SharedExpert
from MoE.sparse_moe import Config

class TestSharedExpert(unittest.TestCase):
    def setUp(self):
        # 设置基本配置
        self.config = Config(
            hidden_dim=512,    # 隐藏层维度
            expert_num=8,      # 专家数量
            top_k=2,          # 选择前k个专家
            shared_expert=2    # 共享专家数量
        )
        
        # 初始化模型
        self.model = SharedExpert(self.config)
        
        # 测试数据维度
        self.batch_size = 4
        self.seq_len = 16

    def test_forward_pass(self):
        # 测试基本的前向传播
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        output, router_logits = self.model(x)
        
        # 检查输出维度
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_dim))
        self.assertEqual(router_logits.shape, (self.batch_size * self.seq_len, self.config.expert_num))

    def test_shared_expert_computation(self):
        # 测试共享专家的计算
        x = torch.randn(1, 1, self.config.hidden_dim)
        
        # 获取每个共享专家的输出
        shared_outputs = []
        for expert in self.model.shared_experts:
            expert_output = expert(x)
            shared_outputs.append(expert_output)
            
            # 检查每个专家的输出维度
            self.assertEqual(expert_output.shape, (1, 1, self.config.hidden_dim))
        
        # 检查共享专家的输出是否不同
        for i in range(len(shared_outputs)):
            for j in range(i + 1, len(shared_outputs)):
                self.assertFalse(torch.allclose(shared_outputs[i], shared_outputs[j], atol=1e-6))

    def test_shared_and_routed_combination(self):
        # 测试共享专家和路由专家的组合
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        
        # 获取共享专家的输出
        shared_outputs = [expert(x) for expert in self.model.shared_experts]
        shared_sum = sum(shared_outputs)
        
        # 获取路由专家的输出
        routed_output, _ = self.model.routed_expert_moe(x)
        
        # 获取组合输出
        combined_output, _ = self.model(x)
        
        # 检查维度
        self.assertEqual(shared_sum.shape, combined_output.shape)
        self.assertEqual(routed_output.shape, combined_output.shape)
        
        # 验证组合逻辑
        expected_output = shared_sum + routed_output
        self.assertTrue(torch.allclose(combined_output, expected_output, atol=1e-6))

    def test_single_sample(self):
        # 测试单个样本的处理
        x = torch.randn(1, 1, self.config.hidden_dim)
        output, router_logits = self.model(x)
        
        # 检查输出维度
        self.assertEqual(output.shape, (1, 1, self.config.hidden_dim))
        self.assertEqual(router_logits.shape, (1, self.config.expert_num))

    def test_shared_expert_count(self):
        # 测试共享专家的数量
        self.assertEqual(len(self.model.shared_experts), self.config.shared_expert)

    def test_output_range(self):
        # 测试输出值的范围
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        output, _ = self.model(x)
        
        # 检查输出是否在合理范围内
        self.assertTrue(torch.isfinite(output).all())  # 检查是否有无穷大或NaN值
        
        # 检查输出的统计特性
        mean = output.mean().item()
        std = output.std().item()
        self.assertTrue(-10 < mean < 10)  # 均值应该在合理范围内
        self.assertTrue(0 < std < 10)     # 标准差应该为正且在合理范围内

if __name__ == '__main__':
    unittest.main() 