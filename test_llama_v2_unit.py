#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为llama_v2模块提供单元测试
测试包括:
1. 模型架构 (model.py)
2. 分词器 (tokenizer.py)
3. 文本生成 (generation.py)
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_v2.tokenizer import Tokenizer
from llama_v2.model import V2ModelArgs, Transformer, RMSNorm, RotaryEmbedding, Attention
from llama_v2.generation import sample_top_p


class TestLlamaV2Model(unittest.TestCase):
    """测试llama_v2.model模块中的组件"""
    
    def setUp(self):
        """设置测试环境"""
        self.model_args = V2ModelArgs(
            dim=512,              # 小型测试模型
            n_layers=2,           # 减少层数以加速测试
            n_heads=8,            # 注意力头
            n_kv_heads=4,         # KV头
            vocab_size=32000,     # 词汇表大小
            norm_eps=1e-5,
            max_batch_size=2,     # 小批量以加速测试
            max_seq_len=32        # 短序列以加速测试
        )
        
        # 使用CPU用于测试
        self.device = torch.device('cpu')
    
    def test_rms_norm(self):
        """测试RMSNorm层"""
        batch_size = 2
        seq_len = 16
        hidden_dim = 512
        
        # 创建RMSNorm实例
        rms_norm = RMSNorm(hidden_dim)
        
        # 创建随机输入
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 测试前向传播
        output = rms_norm(x)
        
        # 检查输出形状
        self.assertEqual(output.shape, x.shape)
        
        # 检查归一化效果（均值接近0，方差接近1）
        mean = output.mean().item()
        std = output.std().item()
        self.assertTrue(-0.1 < mean < 0.1)  # 均值应接近0
        self.assertTrue(0.9 < std < 1.1)    # 经过缩放后标准差应该接近1
    
    def test_rotary_embedding(self):
        """测试旋转位置编码"""
        # 测试预计算频率
        dim = 128
        seq_len = 32
        freqs_cis = RotaryEmbedding.precompute_freqs_cis(dim // 2, seq_len)
        
        # 检查形状
        self.assertEqual(freqs_cis.shape, (seq_len, dim // 4))
        
        # 测试应用旋转编码
        batch_size = 2
        n_heads = 4
        head_dim = dim // n_heads
        
        # 创建模拟查询和键
        xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
        xk = torch.randn(batch_size, seq_len, n_heads, head_dim)
        
        # 应用旋转编码
        xq_out, xk_out = RotaryEmbedding.apply_rotary_emb(xq, xk, freqs_cis)
        
        # 检查输出形状
        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)
    
    def test_attention_forward(self):
        """测试注意力机制的前向传播"""
        # 创建注意力模块
        attention = Attention(self.model_args)
        
        # 创建输入
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, self.model_args.dim)
        
        # 创建频率
        freqs_cis = RotaryEmbedding.precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads, seq_len * 2)
        
        # 创建掩码
        mask = torch.zeros(1, 1, seq_len, seq_len)
        
        # 前向传播
        output = attention(x, 0, freqs_cis[:seq_len], mask)
        
        # 检查输出形状
        self.assertEqual(output.shape, x.shape)
    
    def test_transformer_init(self):
        """测试Transformer初始化"""
        # 初始化Transformer
        transformer = Transformer(self.model_args)
        
        # 检查层数
        self.assertEqual(len(transformer.layers), self.model_args.n_layers)
        
        # 检查频率计算
        self.assertIsNotNone(transformer.freqs_cis)
    
    def test_transformer_forward(self):
        """测试Transformer前向传播"""
        # 初始化Transformer
        transformer = Transformer(self.model_args)
        
        # 创建输入token
        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, self.model_args.vocab_size, (batch_size, seq_len))
        
        # 前向传播
        with torch.inference_mode():
            output = transformer(tokens, 0)
        
        # 检查输出形状
        self.assertEqual(output.shape, (batch_size, seq_len, self.model_args.vocab_size))


class TestLlamaV2Tokenizer(unittest.TestCase):
    """测试llama_v2.tokenizer模块"""
    
    def setUp(self):
        """设置测试环境"""
        # 由于我们不能直接测试实际的tokenizer（需要模型文件），
        # 这里我们会使用mock对象
        self.sentencepiece_mock = MagicMock()
        self.sentencepiece_mock.vocab_size.return_value = 32000
        self.sentencepiece_mock.bos_id.return_value = 1
        self.sentencepiece_mock.eos_id.return_value = 2
        self.sentencepiece_mock.pad_id.return_value = 0
        
    @patch('llama_v2.tokenizer.SentencePieceProcessor')
    def test_tokenizer_init(self, mock_sentencepiece):
        """测试Tokenizer初始化"""
        # 设置mock
        mock_sentencepiece.return_value = self.sentencepiece_mock
        
        # 使用临时路径
        with tempfile.NamedTemporaryFile() as temp:
            # 初始化Tokenizer
            tokenizer = Tokenizer(temp.name)
            
            # 检查基本属性
            self.assertEqual(tokenizer.n_words, 32000)
            self.assertEqual(tokenizer.bos_id, 1)
            self.assertEqual(tokenizer.eos_id, 2)
            self.assertEqual(tokenizer.pad_id, 0)
    
    @patch('llama_v2.tokenizer.SentencePieceProcessor')
    def test_tokenizer_encode(self, mock_sentencepiece):
        """测试编码方法"""
        # 设置mock
        mock_sentencepiece.return_value = self.sentencepiece_mock
        self.sentencepiece_mock.encode.return_value = [100, 200, 300]
        
        # 使用临时路径
        with tempfile.NamedTemporaryFile() as temp:
            # 初始化Tokenizer
            tokenizer = Tokenizer(temp.name)
            
            # 测试编码
            tokens = tokenizer.encode("测试文本", bos=True, eos=True)
            
            # 检查结果
            self.assertEqual(tokens, [1, 100, 200, 300, 2])  # bos + tokens + eos
            
            # 测试不带bos和eos
            self.sentencepiece_mock.encode.return_value = [400, 500]
            tokens = tokenizer.encode("另一个测试", bos=False, eos=False)
            self.assertEqual(tokens, [400, 500])  # 仅tokens
    
    @patch('llama_v2.tokenizer.SentencePieceProcessor')
    def test_tokenizer_decode(self, mock_sentencepiece):
        """测试解码方法"""
        # 设置mock
        mock_sentencepiece.return_value = self.sentencepiece_mock
        self.sentencepiece_mock.decode.return_value = "解码后的文本"
        
        # 使用临时路径
        with tempfile.NamedTemporaryFile() as temp:
            # 初始化Tokenizer
            tokenizer = Tokenizer(temp.name)
            
            # 测试解码
            text = tokenizer.decode([100, 200, 300])
            
            # 检查结果
            self.assertEqual(text, "解码后的文本")
            self.sentencepiece_mock.decode.assert_called_with([100, 200, 300])


class TestLlamaV2Generation(unittest.TestCase):
    """测试llama_v2.generation模块"""
    
    def test_sample_top_p(self):
        """测试top-p采样函数"""
        # 创建概率分布
        probs = torch.tensor([
            [0.5, 0.2, 0.1, 0.1, 0.05, 0.05],  # 第一个样本
            [0.1, 0.1, 0.2, 0.5, 0.05, 0.05]   # 第二个样本
        ])
        
        # 使用不同的p值测试
        for p in [0.5, 0.7, 0.9]:
            next_token = sample_top_p(probs, p)
            
            # 检查输出形状
            self.assertEqual(next_token.shape, (2, 1))
            
            # 检查输出值范围
            self.assertTrue(torch.all(next_token >= 0))
            self.assertTrue(torch.all(next_token < probs.size(1)))
    
    def test_text_generation_workflow(self):
        """测试文本生成的整体工作流程（使用mock）"""
        # 这个测试演示如何测试生成过程的逻辑，而无需实际模型
        # 实际使用时，可能需要更复杂的模拟
        
        # 创建模拟模型和tokenizer
        model_mock = MagicMock()
        model_mock.params.max_seq_len = 2048
        model_mock.params.max_batch_size = 4
        model_mock.forward.return_value = torch.randn(2, 1, 32000)  # 模拟logits输出
        
        tokenizer_mock = MagicMock()
        tokenizer_mock.pad_id = 0
        tokenizer_mock.eos_id = 2
        tokenizer_mock.encode.return_value = [1, 100, 200]
        tokenizer_mock.decode.return_value = "生成的文本"
        
        # 我们不能直接测试Llama类，因为它需要实际的模型和分词器
        # 但我们可以测试sample_top_p函数和基本生成逻辑
        
        # 测试sample_top_p函数的行为
        probs = torch.softmax(torch.randn(2, 32000), dim=-1)
        top_p = 0.9
        next_token = sample_top_p(probs, top_p)
        self.assertEqual(next_token.shape, (2, 1))


if __name__ == "__main__":
    unittest.main() 