#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaMA v2 模型演示脚本
此脚本演示了如何加载和使用LLaMA v2模型进行文本生成
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path
from typing import List, Optional

from llama_v2.model import V2ModelArgs, Transformer
from llama_v2.tokenizer import Tokenizer
from llama_v2.generation import Llama


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLaMA v2 模型演示")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="包含模型检查点的目录路径"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="分词器模型文件路径"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="生成时的温度参数 (0.0-1.0)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="生成时的top-p采样参数 (0.0-1.0)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="模型处理的最大序列长度"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="最大批处理大小"
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=256,
        help="生成文本的最大长度"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["completion", "chat"],
        default="completion",
        help="运行模式：文本补全(completion)或聊天(chat)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["你好，请问"],
        help="用于生成的提示文本"
    )

    return parser.parse_args()


def completion_mode(llama: Llama, args):
    """运行文本补全模式"""
    print("\n===== 文本补全模式 =====")
    print("提示语:")
    for i, prompt in enumerate(args.prompts):
        print(f"[{i+1}] {prompt}")
    
    print("\n正在生成...")
    start_time = time.time()
    results = llama.text_completion(
        prompts=args.prompts,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p
    )
    end_time = time.time()
    
    print("\n生成结果:")
    for i, (prompt, result) in enumerate(zip(args.prompts, results)):
        print(f"\n提示 [{i+1}]: {prompt}")
        print(f"生成: {result['generation']}")
        print("-" * 50)
    
    print(f"\n生成耗时: {end_time - start_time:.2f} 秒")


def chat_mode(llama: Llama, args):
    """运行聊天模式"""
    print("\n===== 聊天模式 =====")
    print("系统提示: 你是一个有用的人工智能助手。")
    
    # 构建对话
    dialogs = []
    for prompt in args.prompts:
        dialogs.append([
            {"role": "system", "content": "你是一个有用的人工智能助手。"},
            {"role": "user", "content": prompt}
        ])
    
    print("\n用户提示:")
    for i, prompt in enumerate(args.prompts):
        print(f"[{i+1}] {prompt}")
    
    print("\n正在生成回复...")
    start_time = time.time()
    results = llama.chat_completion(
        dialogs=dialogs,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p
    )
    end_time = time.time()
    
    print("\n对话结果:")
    for i, (prompt, result) in enumerate(zip(args.prompts, results)):
        print(f"\n用户 [{i+1}]: {prompt}")
        print(f"助手: {result['generation']['content']}")
        print("-" * 50)
    
    print(f"\n生成耗时: {end_time - start_time:.2f} 秒")


def interactive_chat_mode(llama: Llama, args):
    """运行交互式聊天模式"""
    print("\n===== 交互式聊天模式 =====")
    print("系统提示: 你是一个有用的人工智能助手。")
    print("输入 'exit', 'quit', 或 'q' 结束对话。")
    
    # 初始化对话历史
    dialog = [
        {"role": "system", "content": "你是一个有用的人工智能助手。"}
    ]
    
    while True:
        # 获取用户输入
        user_input = input("\n用户: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("\n对话结束。")
            break
        
        # 添加用户输入到对话历史
        dialog.append({"role": "user", "content": user_input})
        
        # 准备对话格式
        current_dialog = list(dialog)  # 复制对话历史
        
        print("助手正在思考...")
        start_time = time.time()
        
        # 生成回复
        result = llama.chat_completion(
            dialogs=[current_dialog],
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        end_time = time.time()
        
        # 获取并显示回复
        assistant_response = result[0]["generation"]["content"]
        print(f"助手: {assistant_response}")
        print(f"(生成耗时: {end_time - start_time:.2f} 秒)")
        
        # 添加助手回复到对话历史
        dialog.append({"role": "assistant", "content": assistant_response})


def main():
    """主函数"""
    args = parse_args()
    
    print(f"加载模型中... 检查点目录: {args.ckpt_dir}")
    print(f"分词器路径: {args.tokenizer_path}")
    
    # 检查文件存在
    if not os.path.exists(args.ckpt_dir):
        print(f"错误: 检查点目录不存在: {args.ckpt_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.tokenizer_path):
        print(f"错误: 分词器文件不存在: {args.tokenizer_path}")
        sys.exit(1)
    
    # 检查是否有GPU可用
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 未检测到GPU，使用CPU可能会很慢。")
    
    try:
        # 构建LLaMA实例
        llama = Llama.build(
            ckpt_dir=args.ckpt_dir,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size
        )
        
        # 根据模式运行
        if args.mode == "completion":
            completion_mode(llama, args)
        elif args.mode == "chat":
            if len(args.prompts) == 1 and args.prompts[0] == "interactive":
                interactive_chat_mode(llama, args)
            else:
                chat_mode(llama, args)
                
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 