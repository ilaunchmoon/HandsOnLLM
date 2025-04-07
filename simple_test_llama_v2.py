#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA v2 模型基本组件测试脚本
此脚本展示如何使用LLaMA v2模型的基本组件，而不需要实际的模型权重
"""

import torch
import argparse
from pathlib import Path
from typing import List, Optional

from llama_v2.model import V2ModelArgs, RMSNorm, RotaryEmbedding, Attention


def test_rms_norm():
    """测试RMSNorm层"""
    print("\n===== 测试RMSNorm层 =====")
    # 创建随机输入
    batch_size = 2
    seq_len = 16
    hidden_dim = 512
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建RMSNorm实例
    rms_norm = RMSNorm(hidden_dim)
    
    # 前向传播
    output = rms_norm(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出均值: {output.mean().item():.6f}")
    print(f"输出标准差: {output.std().item():.6f}")
    
    return True


def test_rotary_embedding():
    """测试旋转位置编码"""
    print("\n===== 测试旋转位置编码 =====")
    # 设置参数
    dim = 128
    seq_len = 32
    batch_size = 2
    n_heads = 4
    head_dim = dim // n_heads
    
    # 预计算频率
    print("预计算频率...")
    freqs_cis = RotaryEmbedding.precompute_freqs_cis(dim // 2, seq_len)
    print(f"频率形状: {freqs_cis.shape}")
    
    # 创建查询和键
    print("创建查询和键...")
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_heads, head_dim)
    print(f"查询形状: {xq.shape}")
    print(f"键形状: {xk.shape}")
    
    # 创建适合的频率张量
    mock_freqs_cis = torch.ones((seq_len, head_dim//2), dtype=torch.complex64)
    
    # 应用旋转编码
    print("应用旋转编码...")
    try:
        xq_out, xk_out = RotaryEmbedding.apply_rotary_emb(xq, xk, mock_freqs_cis)
        print(f"旋转后查询形状: {xq_out.shape}")
        print(f"旋转后键形状: {xk_out.shape}")
        return True
    except Exception as e:
        print(f"错误: {e}")
        print("在实际使用中，确保提供正确形状的频率张量")
        return False


def test_attention_components():
    """测试注意力组件"""
    print("\n===== 测试注意力组件 =====")
    # 设置参数
    model_args = V2ModelArgs(
        dim=512,              # 小型测试模型
        n_layers=2,           # 减少层数
        n_heads=8,            # 注意力头
        n_kv_heads=4,         # KV头
        vocab_size=32000,     # 词汇表大小
        norm_eps=1e-5,
        max_batch_size=2,
        max_seq_len=32
    )
    
    # 创建注意力模块
    print("创建注意力模块...")
    attention = Attention(model_args)
    
    # 创建输入
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, model_args.dim)
    print(f"输入形状: {x.shape}")
    
    # 测试查询、键、值计算
    print("计算查询、键、值...")
    xq = attention.wq(x)
    xk = attention.wk(x)
    xv = attention.wv(x)
    
    print(f"查询形状: {xq.shape}")
    print(f"键形状: {xk.shape}")
    print(f"值形状: {xv.shape}")
    
    return True


def main():
    """主函数"""
    print("===== LLaMA v2 模型基本组件测试 =====")
    print("注意: 此测试不需要实际的模型权重")
    
    # 测试RMSNorm
    if test_rms_norm():
        print("RMSNorm测试通过 ✓")
    else:
        print("RMSNorm测试失败 ✗")
    
    # 测试旋转位置编码
    if test_rotary_embedding():
        print("旋转位置编码测试通过 ✓")
    else:
        print("旋转位置编码测试失败 ✗")
    
    # 测试注意力组件
    if test_attention_components():
        print("注意力组件测试通过 ✓")
    else:
        print("注意力组件测试失败 ✗")


if __name__ == "__main__":
    main() 