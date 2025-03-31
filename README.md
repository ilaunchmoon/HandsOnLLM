# Hands on LLM

For the warehouse is mainly to record some open source large language model learning notes

# 手动实现LLaMA V1模型

这个项目是一个简单的LLaMA V1模型的实现。

## 目录结构

```
.
├── llama_v1/                # LLaMA V1模型实现
│   ├── __init__.py         
│   ├── tokenizer.py         # 分词器实现
│   ├── model.py             # 模型架构实现
│   └── generation.py        # 文本生成逻辑实现
├── test_llama.py            # 简单的测试脚本
├── test_llama_complete.py   # 完整的测试脚本（带参数解析和错误处理）
└── README.md                # 项目说明文档
```

## 依赖安装

```bash
pip install torch sentencepiece
```

## 使用方法

### 简单测试

```bash
python test_llama.py
```

这将运行一个简单的演示，展示如何使用该模型，但不会实际加载或运行模型。

### 完整测试

```bash
python test_llama_complete.py --model_path PATH_TO_MODEL --tokenizer_path PATH_TO_TOKENIZER
```

参数说明：

- `--model_path`: LLaMA模型权重路径 (.pth 文件)
- `--tokenizer_path`: SentencePiece tokenizer模型路径 (.model 文件)
- `--prompts`: 用于测试的提示文本列表，默认为"今天天气真好，我想去"
- `--max_gen_len`: 生成文本的最大长度，默认为50
- `--temperature`: 生成采样的温度，默认为0.8
- `--top_p`: nucleus采样的概率阈值，默认为0.95

示例：

```bash
python test_llama_complete.py --model_path ./weights/llama_v1.pth --tokenizer_path ./tokenizer/tokenizer.model --prompts "你好，请问" "如何学习人工智能" --max_gen_len 100
```

## 注意事项

- 这个实现需要预训练的LLaMA模型权重和SentencePiece tokenizer模型才能实际运行
- 请确保您有权访问和使用这些资源
- 代码中已经修复了一些已知的错误，如果发现更多问题，请提交issue

