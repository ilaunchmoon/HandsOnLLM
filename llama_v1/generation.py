import json
import os
import sys
import time
import torch
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

from llama_v1.tokenizer import Tokenizer
from llama_v1.model import Transformer


def sample_top_p(probs:torch.Tensor, top_p:float)->float:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)      # 将probs中的元素进行降序排列, 返回降序排列后的结果, 以及排序前各元素的索引
    probs_sum = torch.cumsum(probs_sort, dim=-1)                            # 将降序后的元素逐个累加, 如[0.4, 0.3, 0.2, 0.1] --> [0.4, 0.7, 0.9, 1.0]
    mask = probs_sum - probs_sort > top_p                                   # 构建掩码矩阵: 若累积后概率向量元素减去排序后的对应位置元素还大于top_p, 则它需要被掩码为0, 便于后期取前top_p个
    probs_sort[mask] = 0.0                                                  # 掩码操作
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))                   # 掩码操作后, 归一化
    next_token = torch.multinomial(probs_sort, num_samples=1)               # 归一化, 再依赖于归一化后的概率分布随机采样一个样本
    next_token = torch.gather(probs_idx, -1, next_token)                    # 通过原始probs的各元素索引, 在next_token的最后一个维度上收集元素, 并返回给next_token
    return next_token




class LLamaV1Generator:
    def __init__(self, model:Transformer, token_model:Tokenizer) -> None:
        self.model = model
        self.token_model = token_model
    
    def generate(self,
                prompts:List[str],                  # 输入的提示词
                max_gen_len:int,                    # 最大生成eq的长度
                temperature:float=0.8,              # 温度设置, 用户控制生成token的随机性强度
                top_p:float=0.95                    # token by token过程中，用于生成下一个token的时概率分布, 获取该概率分布的前top个概率部分, 然后再归一化
                )->List[str]:
        batch_size:int = len(prompts)               # 提示词的个数
        params = self.model.args                    # 解码器模型的参数配置信息

        # 验证当前输入prompts的长度小于等于模型最大处理长度, 如果不是小于等于, 则会抛出以元组形式(batch_size, params.max_batch_size)异常信息 
        assert batch_size <= params.max_batch_size, (batch_size, params.max_batch_size)     

        # 调用token模型的编码器对输入的prompt进行编码操作
        # prompt_token其实就是输入进来的prompts经过编码之后的token id
        prompt_token = [self.token_model.encode(x, bos=True, eos=False) for x in prompts]

        # 获取提示词编码之后最小的token序列长度和最大的token序列长度
        min_prompt_size = min([len(t) for t in prompt_token])
        max_prompt_size = max([len(t) for t in prompt_token])

        # 获取当前能够处理的最大序列长度: 在模型能够处理的最大长度与输入最大长度加最大提示词长度的和 之间取最小值
        # 主要目的就是为了防止模型处理超过它最大处理长度的部分
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        
        # 预先生成一个存放token id的张量, 先将全部的原始都设为pad_id
        tokens = torch.full((batch_size, total_len), self.token_model.pad_id).long()
        for k, t in enumerate(prompt_token):
            tokens[k,:len(t)] = torch.tensor(t).long()      # 将tokens的第k个批次中的后len(t)个位置都设置为prompt_token的索引, 这里的t其实就是token的id
        
        input_text_mask = tokens != self.token_model.pad_id # 如果tokens不是pad_id, 则设置为True, 否则设置的False, 用于构建一个掩码矩阵
        start_pos = min_prompt_size                         # 将开始生成的位置设置为最小的prompt长度
        prev_pos = 0                                        # token by token的生成过程中, 当前生成的token的前一个token的位置
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)      # 在之前预测的token基础上预测当前位置的token
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)                 # 下一个token id的位置, 使用采样前top_p的方式来预测下一token的位置
            else:   
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)             # 展平
            next_token =  torch.where(
                input_text_mask[:,cur_pos],                 # 如果当前cur_pos位置为真, 则说明没有掩码, 则获取token中的当前cur_pos位置上的token, 否则直接获取next_token
                tokens[:cur_pos],                           # 
                next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos                              # 更新prev_pos的位置

        
        # 进行解码token id所对应的分词token
        decode = []
        for i, t in enumerate(tokens.tolist()):
            t = t[:len(prompt_token[i] + max_gen_len)]      # 最大生成长度以外的直接裁剪
            try:
                t = t[:, t.index(self.token_model.eos_id)]  
            except ValueError:
                pass
            decode.append(self.token_model.decode(t))
        return decode
    


                

