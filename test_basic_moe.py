import torch
import unittest
from MoE.basic_moe import BasicMoe, BasicExpert

class TestBasicMoe(unittest.TestCase):
    def setUp(self):
        # 设置基本配置
        self.feat_in = 256
        self.hidden_dim = 512
        self.feat_out = 128
        self.num_expert = 4
        self.batch_size = 2
        self.seq_len = 10
        
        # 初始化模型
        self.model = BasicMoe(
            feat_in=self.feat_in,
            hidden_dim=self.hidden_dim,
            feat_out=self.feat_out,
            num_expert=self.num_expert
        )

    def test_forward_pass(self):
        # 测试基本的前向传播
        x = torch.randn(self.batch_size, self.seq_len, self.feat_in)
        output = self.model(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.feat_out))

    def test_expert_output(self):
        # 测试每个专家的输出
        x = torch.randn(self.batch_size, self.seq_len, self.feat_in)
        
        # 获取门控权重
        gate = self.model.gate(x)
        expert_weight = torch.softmax(gate, dim=-1)
        
        # 检查门控权重的形状
        self.assertEqual(gate.shape, (self.batch_size, self.seq_len, self.num_expert))
        self.assertEqual(expert_weight.shape, (self.batch_size, self.seq_len, self.num_expert))
        
        # 检查权重和是否接近1
        weight_sum = expert_weight.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6))

    def test_single_sample(self):
        # 测试单个样本的处理
        x = torch.randn(1, 1, self.feat_in)
        output = self.model(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, (1, 1, self.feat_out))

    def test_expert_computation(self):
        # 测试专家计算的正确性
        x = torch.randn(1, 1, self.feat_in)
        
        # 获取每个专家的输出
        expert_outputs = []
        for expert in self.model.expert_net:
            expert_output = expert(x)
            expert_outputs.append(expert_output)
            
            # 检查每个专家的输出维度
            self.assertEqual(expert_output.shape, (1, 1, self.feat_out))
        
        # 检查所有专家的输出是否不同
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                self.assertFalse(torch.allclose(expert_outputs[i], expert_outputs[j], atol=1e-6))

if __name__ == '__main__':
    unittest.main() 