import torch
import unittest
from MoE.sparse_moe import SparseMoe, Config, Router

class TestSparseMoe(unittest.TestCase):
    def setUp(self):
        # 设置基本配置
        self.config = Config(
            hidden_dim=512,    # 隐藏层维度
            expert_num=8,      # 专家数量
            top_k=2,          # 选择前k个专家
            shared_expert=2    # 共享专家数量
        )
        
        # 初始化模型
        self.model = SparseMoe(self.config)
        
        # 测试数据维度
        self.batch_size = 4
        self.seq_len = 16

    def test_router(self):
        # 测试路由器功能
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        router = Router(self.config)
        
        # 获取路由结果
        router_logits, router_weight, top_k_indices, expert_mask = router(x.view(-1, self.config.hidden_dim))
        
        # 检查维度
        self.assertEqual(router_logits.shape, (self.batch_size * self.seq_len, self.config.expert_num))
        self.assertEqual(router_weight.shape, (self.batch_size * self.seq_len, self.config.top_k))
        self.assertEqual(top_k_indices.shape, (self.batch_size * self.seq_len, self.config.top_k))
        self.assertEqual(expert_mask.shape, (self.config.expert_num, self.config.top_k, self.batch_size * self.seq_len))
        
        # 检查路由权重是否归一化
        weight_sum = router_weight.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6))
        
        # 检查top_k索引是否在合法范围内
        self.assertTrue((top_k_indices >= 0).all())
        self.assertTrue((top_k_indices < self.config.expert_num).all())

    def test_forward_pass(self):
        # 测试基本的前向传播
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        output, router_logits = self.model(x)
        
        # 检查输出维度
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_dim))
        self.assertEqual(router_logits.shape, (self.batch_size * self.seq_len, self.config.expert_num))

    def test_expert_selection(self):
        # 测试专家选择机制
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        
        # 获取路由结果
        router_logits, router_weight, top_k_indices, expert_mask = self.model.router(x.view(-1, self.config.hidden_dim))
        
        # 检查是否只选择了top_k个专家
        selected_experts_count = expert_mask.sum(dim=1)  # 每个位置选中的专家数量
        self.assertTrue(torch.all(selected_experts_count <= self.config.top_k))

    def test_single_sample(self):
        # 测试单个样本的处理
        x = torch.randn(1, 1, self.config.hidden_dim)
        output, router_logits = self.model(x)
        
        # 检查输出维度
        self.assertEqual(output.shape, (1, 1, self.config.hidden_dim))
        self.assertEqual(router_logits.shape, (1, self.config.expert_num))

    def test_expert_computation(self):
        # 测试专家计算的正确性
        x = torch.randn(1, 1, self.config.hidden_dim)
        
        # 获取每个专家的输出
        expert_outputs = []
        for expert in self.model.experts:
            expert_output = expert(x)
            expert_outputs.append(expert_output)
            
            # 检查每个专家的输出维度
            self.assertEqual(expert_output.shape, (1, 1, self.config.hidden_dim))
        
        # 检查所有专家的输出是否不同
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                self.assertFalse(torch.allclose(expert_outputs[i], expert_outputs[j], atol=1e-6))

    def test_shared_experts(self):
        # 测试共享专家的功能
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_dim)
        
        # 获取路由结果
        router_logits, router_weight, top_k_indices, expert_mask = self.model.router(x.view(-1, self.config.hidden_dim))
        
        # 检查是否有共享专家的使用
        expert_usage = expert_mask.sum(dim=(1, 2))  # 统计每个专家被使用的次数
        self.assertTrue((expert_usage > 0).any())  # 至少有一个专家被使用

if __name__ == '__main__':
    unittest.main() 