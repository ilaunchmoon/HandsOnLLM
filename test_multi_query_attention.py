#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试MultiQueryAttention模块的功能完整性
"""

import unittest
import torch
import torch.nn as nn
from Attention.GoupQueryAttention import GroupQueryAttention

class TestMultiQueryAttention(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.hidden_dim = 512
        self.num_heads = 8
        self.num_kv_groups = 2
        self.dropout_rate = 0.1
        self.batch_size = 2
        self.seq_len = 16
        
        # 创建GroupQueryAttention实例
        self.attention = GroupQueryAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_kv_groups=self.num_kv_groups,
            dropout_rate=self.dropout_rate
        )
        
        # 创建测试输入
        self.x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        self.mask = torch.ones(self.batch_size, 1, 1, 1, self.seq_len)
    
    def test_initialization(self):
        """测试初始化是否正确"""
        # 检查投影层
        self.assertIsInstance(self.attention.q_proj, nn.Linear)
        self.assertIsInstance(self.attention.k_proj, nn.Linear)
        self.assertIsInstance(self.attention.v_proj, nn.Linear)
        self.assertIsInstance(self.attention.out_proj, nn.Linear)
        
        # 检查参数
        self.assertEqual(self.attention.num_heads, self.num_heads)
        self.assertEqual(self.attention.num_kv_groups, self.num_kv_groups)
        self.assertEqual(self.attention.head_dim, self.hidden_dim // self.num_heads)
        self.assertEqual(self.attention.heads_per_group, self.num_heads // self.num_kv_groups)
    
    def test_forward_shape(self):
        """测试前向传播的输出形状"""
        # 执行前向传播
        output = self.attention(self.x, self.mask)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
    
    def test_attention_weights(self):
        """测试注意力权重的计算"""
        # 执行前向传播
        output = self.attention(self.x, self.mask)
        
        # 检查输出是否包含NaN
        self.assertFalse(torch.isnan(output).any())
        
        # 检查输出是否包含inf
        self.assertFalse(torch.isinf(output).any())
    
    def test_masking(self):
        """测试掩码功能"""
        # 创建一个掩码，将部分位置设为0
        mask = torch.ones(self.batch_size, 1, 1, 1, self.seq_len)
        mask[:, :, :, :, :self.seq_len//2] = 0
        
        # 执行前向传播
        output = self.attention(self.x, mask)
        
        # 检查输出是否有效
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        # 创建优化器
        optimizer = torch.optim.SGD(self.attention.parameters(), lr=0.01)
        
        # 执行前向传播和反向传播
        output = self.attention(self.x, self.mask)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在
        for name, param in self.attention.named_parameters():
            self.assertIsNotNone(param.grad, f"参数 {name} 没有梯度")
            self.assertFalse(torch.isnan(param.grad).any(), f"参数 {name} 的梯度包含NaN")
            self.assertFalse(torch.isinf(param.grad).any(), f"参数 {name} 的梯度包含inf")

if __name__ == '__main__':
    unittest.main() 