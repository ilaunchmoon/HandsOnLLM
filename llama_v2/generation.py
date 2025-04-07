import json
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from llama_v2.model import V2ModelArgs, Transformer
from llama_v2.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # 非必需
    logprobs: List[float]  # 非必需


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # 非必需
    logprobs: List[float]  # 非必需


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
    ) -> "Llama":
        """
        通过初始化和加载预训练模型来创建Llama实例

        参数:
            ckpt_dir (str): 包含检查点文件的目录路径
            tokenizer_path (str): 分词器文件的路径
            max_seq_len (int): 输入文本的最大序列长度
            max_batch_size (int): 推理时的最大批处理大小

        返回:
            Llama: 加载了模型和分词器的Llama类实例

        异常:
            AssertionError: 如果指定目录中没有检查点文件

        注意:
            此方法设置设备为CUDA, 并加载预训练模型和分词器
        """
        # 设置设备
        torch.cuda.set_device(0)

        # 设置随机种子以确保可重复性
        torch.manual_seed(1)

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"在{ckpt_dir}中未找到检查点文件"
        
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: V2ModelArgs = V2ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"加载完成，耗时{time.time() - start_time:.2f}秒")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        基于提供的提示生成文本序列

        参数:
            prompt_tokens (List[List[int]]): 已分词的提示列表，每个提示表示为整数列表
            max_gen_len (int): 生成文本序列的最大长度
            temperature (float, optional): 控制采样随机性的温度值, 默认为0.6
            top_p (float, optional): 核采样的概率阈值, 默认为0.9
            logprobs (bool, optional): 是否计算词元的对数概率, 默认为False
            echo (bool, optional): 是否在生成的输出中包含提示词元, 默认为False

        返回:
            Tuple[List[List[int]], Optional[List[List[float]]]]: 包含生成的词元序列的元组, 如果logprobs为True, 则包含相应的词元对数概率

        注意:
            此方法使用提供的提示作为生成文本的基础。它使用核采样来产生具有可控随机性的文本
            如果logprobs为True, 则计算每个生成词元的对数概率
        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # 仅在提示已经生成时替换词元
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # 截断到最大生成长度
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # 如果有结束词元则截断
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        使用语言生成模型为一系列提示执行文本补全

        参数:
            prompts (List[str]): 用于补全的文本提示列表
            temperature (float, optional): 控制采样随机性的温度值, 默认为0.6
            top_p (float, optional): 核采样的概率阈值, 默认为0.9
            max_gen_len (Optional[int], optional): 生成补全序列的最大长度, 如果未提供, 则设置为模型的最大序列长度减1,
            logprobs (bool, optional): 是否计算词元的对数概率。默认为False
            echo (bool, optional): 是否在生成的输出中包含提示词元。默认为False

        返回:
            List[CompletionPrediction]: 补全预测列表，每个包含生成的文本补全

        注意:
            此方法为提供的提示生成文本补全，使用核采样引入可控的随机性
            如果logprobs为True, 则计算每个生成词元的对数概率
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        使用语言生成模型为一系列对话生成助手回复

        参数:
            dialogs (List[Dialog]): 对话列表，每个对话是消息列表
            temperature (float, optional): 控制采样随机性的温度值。默认为0.6
            top_p (float, optional): 核采样的概率阈值。默认为0.9
            max_gen_len (Optional[int], optional): 生成回复序列的最大长度, 如果未提供, 则设置为模型的最大序列长度减1
            logprobs (bool, optional): 是否计算词元的对数概率。默认为False

        返回:
            List[ChatPrediction]: 聊天预测列表，每个包含助手生成的回复

        异常:
            AssertionError: 如果对话中的最后一条消息不是来自用户
            AssertionError: 如果对话角色不按要求的'user'、'assistant'和可选'system'顺序排列

        注意:
            此方法为提供的对话生成助手回复
            它使用核采样在文本生成中引入可控的随机性
            如果logprobs为True, 则计算每个生成词元的对数概率
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "模型仅支持'system'、'user'和'assistant'角色，"
                "从'system'开始，然后是'user'，并交替出现(u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"最后一条消息必须来自用户，得到的是{dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


def sample_top_p(probs, p):
    """
    对概率分布执行top-p采样, 即取大于p的概率

    参数:
        probs (torch.Tensor): 概率分布张量
        p (float): top-p采样的概率阈值

    返回:
        torch.Tensor: 采样的词元索引

    注意:
        Top-p采样选择累积概率质量超过阈值p的最小词元集
        分布基于所选词元重新归一化
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token