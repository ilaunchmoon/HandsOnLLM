# LLaMA v2 模型测试指南

这个文档提供了关于如何运行 LLaMA v2 模型测试单元的说明。

## 测试组件

测试单元包含对以下三个主要组件的测试：

1. **模型架构** (model.py)
   - RMSNorm 层
   - 旋转位置编码
   - 注意力机制
   - Transformer 初始化和前向传播

2. **分词器** (tokenizer.py)
   - 初始化
   - 编码方法
   - 解码方法

3. **文本生成** (generation.py)
   - top-p 采样
   - 文本生成工作流程

## 运行测试

### 前提条件

确保您已经安装了以下依赖：

```bash
pip install torch numpy unittest-mock
```

如果您想要测试实际的 tokenizer，还需要安装：

```bash
pip install sentencepiece
```

### 运行所有测试

要运行所有测试，只需执行：

```bash
python test_llama_v2_unit.py
```

### 运行特定的测试类

要运行特定的测试类，可以使用：

```bash
python -m unittest test_llama_v2_unit.TestLlamaV2Model
python -m unittest test_llama_v2_unit.TestLlamaV2Tokenizer
python -m unittest test_llama_v2_unit.TestLlamaV2Generation
```

### 运行特定的测试方法

要运行特定的测试方法，可以使用：

```bash
python -m unittest test_llama_v2_unit.TestLlamaV2Model.test_rms_norm
```

## 测试说明

- 这些测试使用了 mock 对象来模拟 sentencepiece 分词器，因此不需要实际的分词器模型文件
- 模型测试使用小型配置参数来加速测试过程
- 所有测试都在 CPU 上运行，不需要 GPU
- 测试只检查组件的基本功能，不验证生成文本的质量

## 扩展测试

如果要使用实际的模型权重和分词器进行测试，可以修改测试代码：

1. 在 `setUp` 方法中加载实际的模型权重
2. 使用实际的 sentencepiece 模型初始化分词器
3. 添加测试用例来验证生成文本的质量

例如：

```python
def test_with_real_model(self):
    # 加载实际模型权重和分词器
    model_path = "path/to/model/weights"
    tokenizer_path = "path/to/tokenizer.model"
    
    # 初始化模型和分词器
    model = Transformer.load(model_path)
    tokenizer = Tokenizer(tokenizer_path)
    
    # 运行实际生成测试
    prompts = ["你好，请问"]
    generated_text = generate_text(model, tokenizer, prompts)
    
    # 验证生成的文本
    self.assertTrue(len(generated_text[0]) > len(prompts[0]))
```

## 故障排除

如果在运行测试时遇到问题：

1. 确保所有依赖都已正确安装
2. 检查 Python 版本兼容性（推荐使用 Python 3.8+）
3. 如果使用实际模型，确保有足够的内存和正确的 CUDA 设置 