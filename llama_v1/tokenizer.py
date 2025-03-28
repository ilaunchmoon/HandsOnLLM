import os 
from typing import List
from sentencepiece import SentencePieceProcessor


class Tokenizer():
    def __init__(self, path:str)->None:
        """
            初始化词元化模型sentencepiece
        """
        self.sp_model = SentencePieceProcessor(path)    # 初始化词元化模型
        self.n_words:int = self.sp_model.vocab_size()   # 词汇表大小
        self.bos_id = self.token_model.bos_id()         # 开始标识符token id
        self.eos_id = self.token_model.eos_id()         # 结束标识符token id
        self.pad_id = self.token_model.pad_id()         # 用于填充符的token id

    def encode(self, s:str, bos:bool, eos:bool)->List[int]:
        """
            输入任意token将其编码成token id
            
            s: 输入分词
            bos: 标记是否为序列的初始分词
            eos: 标记是否为序列的结束分词

            返回值: 返回对编码之后token id
        """
        assert type(s) is str                           # 输入的s必须为字符类
        t = self.sp_model.encode(s)                     # 调用词元化模型对s进行编码
        if bos:
            t = [self.bos_id] + t                       # 如果输入的s是起始分词, 则需要在s前面添加一个起始分词对应的token id
        if eos:
            t = t + [self.eos_id]                       # 如果输入的s是最后一个分词, 则需要在后面添加一个结尾分词对应的token id
        return t                                        # 返回编码后的token id

    def decode(self, t:List[int])->str:
        """
            输入任意的token id, 对token id解码成对应分词str
            
            t: 输入的token id
            返回值: 输出解码后的分词str
        """
        return self.sp_model.decode(t)
    


    