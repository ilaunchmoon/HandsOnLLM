# LLaMA v2 模型演示使用说明

这个文档说明如何使用 `demo_llama_v2.py` 脚本来演示 LLaMA v2 模型的功能。

## 前提条件

在运行演示脚本之前，您需要：

1. 下载 LLaMA v2 模型权重文件
2. 下载对应的 SentencePiece 分词器模型文件
3. 安装所需的 Python 依赖项

### 安装依赖

```bash
pip install torch numpy sentencepiece
```

## 模型准备

您需要从 Meta AI 官方渠道获取 LLaMA v2 模型。此过程通常包括：

1. 访问 [Meta AI LLaMA 页面](https://ai.meta.com/llama/)
2. 填写访问请求表单
3. 获得批准后，下载模型权重和分词器

请确保您已经将模型文件放置在适当的位置，并记下：
- 包含模型检查点的目录路径
- 分词器模型文件的路径

## 运行演示

### 基本用法

```bash
python demo_llama_v2.py --ckpt_dir /path/to/model/checkpoints --tokenizer_path /path/to/tokenizer.model
```

这将使用默认参数在文本补全模式下运行模型。

### 命令行参数

演示脚本支持以下命令行参数：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--ckpt_dir` | 包含模型检查点的目录路径 | 必需参数 |
| `--tokenizer_path` | 分词器模型文件路径 | 必需参数 |
| `--temperature` | 生成时的温度参数 (0.0-1.0) | 0.6 |
| `--top_p` | 生成时的 top-p 采样参数 (0.0-1.0) | 0.9 |
| `--max_seq_len` | 模型处理的最大序列长度 | 512 |
| `--max_batch_size` | 最大批处理大小 | 1 |
| `--max_gen_len` | 生成文本的最大长度 | 256 |
| `--mode` | 运行模式：`completion` 或 `chat` | `completion` |
| `--prompts` | 用于生成的提示文本 | "你好，请问" |

### 运行模式

脚本支持两种主要运行模式：

#### 1. 文本补全模式

```bash
python demo_llama_v2.py --ckpt_dir /path/to/checkpoints --tokenizer_path /path/to/tokenizer.model --mode completion --prompts "今天天气真好，我打算" "人工智能的未来发展趋势是"
```

这将生成给定提示的文本补全。

#### 2. 聊天模式

```bash
python demo_llama_v2.py --ckpt_dir /path/to/checkpoints --tokenizer_path /path/to/tokenizer.model --mode chat --prompts "你能介绍一下自己吗？" "如何学习编程？"
```

这将以对话形式生成回复。

#### 3. 交互式聊天模式

```bash
python demo_llama_v2.py --ckpt_dir /path/to/checkpoints --tokenizer_path /path/to/tokenizer.model --mode chat --prompts interactive
```

这将启动一个交互式会话，您可以与模型进行连续对话。

## 示例

### 文本补全示例

```bash
python demo_llama_v2.py --ckpt_dir ./models/llama-2-7b --tokenizer_path ./tokenizer.model --prompts "中国的首都是" --temperature 0.7 --max_gen_len 100
```

### 聊天示例

```bash
python demo_llama_v2.py --ckpt_dir ./models/llama-2-7b-chat --tokenizer_path ./tokenizer.model --mode chat --prompts "解释一下量子计算的基本原理" --temperature 0.8
```

### 多个提示示例

```bash
python demo_llama_v2.py --ckpt_dir ./models/llama-2-7b --tokenizer_path ./tokenizer.model --prompts "写一首关于春天的诗" "介绍一下人工智能的历史" "推荐三本科幻小说"
```

## 注意事项

1. 运行大型模型需要足够的内存和计算资源
2. 使用 GPU 可以显著加快生成速度
3. 降低 `max_seq_len` 和 `max_gen_len` 可以减少内存使用
4. 较高的 `temperature` 值会产生更多样化但可能不太连贯的输出
5. 较低的 `temperature` 值会产生更确定但可能重复的输出

## 故障排除

如果遇到内存错误，请尝试：
- 减小 `max_seq_len` 和 `max_gen_len` 参数
- 使用较小的模型变体（如 7B 而不是 13B）
- 确保有足够的 GPU 内存或切换到具有更多内存的设备

如果遇到生成质量问题，请尝试：
- 调整 `temperature` 和 `top_p` 参数
- 使用更详细或更明确的提示
- 在聊天模式下尝试使用系统提示来引导模型行为 