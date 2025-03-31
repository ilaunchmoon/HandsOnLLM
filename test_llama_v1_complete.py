#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整测试LLaMA V1模型的文本生成功能
包含错误处理和详细的模型加载步骤
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import List, Optional

from llama_v1.tokenizer import Tokenizer
from llama_v1.model import Transformer, V1ModelArgs
from llama_v1.generation import LLamaV1Generator

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试LLaMA V1模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="LLaMA模型权重路径 (.pth 文件)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="SentencePiece tokenizer模型路径 (.model 文件)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["今天天气真好，我想去"],
        help="用于测试的提示文本列表"
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=50,
        help="生成文本的最大长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="生成采样的温度"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="nucleus采样的概率阈值"
    )
    
    return parser.parse_args()

def check_file_exists(file_path: Optional[str], file_type: str) -> bool:
    """检查文件是否存在"""
    if file_path is None:
        print(f"警告: 未提供{file_type}路径")
        return False
    
    path = Path(file_path)
    if not path.exists():
        print(f"错误: {file_type}文件不存在: {file_path}")
        return False
    
    return True

def load_model(model_path: str, model_args: V1ModelArgs) -> Optional[Transformer]:
    """加载模型权重"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        model = Transformer(model_args).to(device)
        print(f"加载模型权重: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 设置为评估模式
        return model
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None

def load_tokenizer(tokenizer_path: str) -> Optional[Tokenizer]:
    """加载tokenizer"""
    try:
        print(f"加载tokenizer: {tokenizer_path}")
        tokenizer = Tokenizer(tokenizer_path)
        return tokenizer
    except Exception as e:
        print(f"加载tokenizer失败: {str(e)}")
        return None

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查文件路径
    model_exists = check_file_exists(args.model_path, "模型权重")
    tokenizer_exists = check_file_exists(args.tokenizer_path, "tokenizer模型")
    
    # 演示模式
    if not model_exists or not tokenizer_exists:
        print("\n进入演示模式 - 不会实际加载或运行模型")
        
        # 配置模型参数
        model_args = V1ModelArgs(
            dim=512,
            n_layers=8,
            n_heads=8,
            vocab_size=32000,
            norm_eps=1e-5,
            max_batch_size=1,
            max_seq_len=512
        )
        
        print("\n要实际运行测试，请提供以下参数:")
        print("python test_llama_complete.py --model_path PATH_TO_MODEL --tokenizer_path PATH_TO_TOKENIZER")
        print("\n例如:")
        print("python test_llama_complete.py --model_path ./weights/llama_v1.pth --tokenizer_path ./tokenizer/tokenizer.model")
        
        return
    
    # 配置模型参数（实际运行时）
    # 注意：在实际使用中，您可能需要从模型配置文件中加载这些参数
    model_args = V1ModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=32000,  # 这个值应该匹配您的tokenizer的词汇表大小
        norm_eps=1e-5,
        max_batch_size=len(args.prompts),  # 根据提示数量设置批次大小
        max_seq_len=512
    )
    
    # 加载模型
    model = load_model(args.model_path, model_args)
    if model is None:
        print("模型加载失败，退出程序")
        return
    
    # 加载tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    if tokenizer is None:
        print("Tokenizer加载失败，退出程序")
        return
    
    # 更新词汇表大小（如果需要）
    if model_args.vocab_size != tokenizer.n_words:
        print(f"警告: 模型词汇表大小 ({model_args.vocab_size}) 与 tokenizer词汇表大小 ({tokenizer.n_words}) 不匹配")
        print("更新模型词汇表大小以匹配tokenizer")
        model_args.vocab_size = tokenizer.n_words
    
    # 初始化生成器
    print("初始化文本生成器...")
    generator = LLamaV1Generator(model, tokenizer)
    
    # 生成文本
    print("\n开始生成文本...")
    try:
        generated_texts = generator.generate(
            prompts=args.prompts,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # 输出结果
        print("\n生成结果:")
        for i, text in enumerate(generated_texts):
            print(f"提示 [{i+1}]: {args.prompts[i]}")
            print(f"生成: {text}")
            print("-" * 50)
    except Exception as e:
        print(f"生成文本时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 