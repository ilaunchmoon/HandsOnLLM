import torch
import torch.nn as nn 
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from .basic_moe import BasicExpert



# 与最基本Moe区别是, SparseMoe具有共享的若干个专家
# 并且专家输出后会选择前top_k个
@dataclass
class Config:
    hidden_dim:int 
    expert_num:int 
    top_k:int 
    shared_expert:int = 2



class Router(nn.Module):
    def __init__(self, config:Config)->None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_dim, config.expert_num)
        self.expert_num = config.expert_num
        self.top_k = config.top_k

    def forward(self, x:torch.Tensor)->Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # (batch_size, seq_len) -->
        # (batch_size * seq_len, expert_num)
        router = self.gate(x)                   
        router_logits = F.softmax(router, dim=-1)

        # 在router_logits的最后一个维度选择前top_k个概率, 以及它们对应的索引
        # (batch_size * seq_len, top_k)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # 上面选择的千top_k个概率一定是不满足概率和为1的, 所以要对选择的top_k个概率重新归一化
        # 此时router_weight就会满足概率之和为1
        router_weight = top_k_logits / top_k_logits.sum(dim=-1, keepdim=True)

        # 做了归一化后, 需要将router_weight的数据类型转变为归一化的数据类型
        # 因为它可能在归一化的过程中改变了数据类型
        router_weight = router_weight.to(x.dtype)


        # 生成expert的mask矩阵
        # 使用上面选择前top_k个权重所对应的索引来生成one-hot向量作为mask矩阵
        # 假如 num_classes = 3 且 top_k_indices为 [[2, 0, 1], [2, 1, 0]]
        # 那么将top_k_indices为转成one-hot向量就是 将top_k_indices为中的每一个元素使用3位0-1编码进行编码
        # 则  expert_mask = [
        # [[0, 0, 1],
        # [1, 0, 0],
        # [0, 1, 0]],
        # [[0, 0, 1],
        # [0, 1, 0],
        # [1, 0, 0]]
        expert_mask = F.one_hot(
            top_k_indices, 
            num_classes=self.expert_num
        )

        # (batch_size * seq_len, top_k, expert_num) --->
        # (expert_num, top_k, batch_size * seq_len)
        expert_mask = expert_mask.permute(2, 1, 0)
    
        # 返回最原始经过gate后的结果                   (batch_size * seq_len, expert_num)
        # 对gate后的结果做归一化后再选择前top_k的概率    (batch_size * seq_len, top_k)
        # 选择前top_k个所对应的索引                    (batch_size * seq_len, top_k)
        # 专家的掩码矩阵                              (expert_num, top_k, batch_size * seq_len)
        return  router, router_weight, top_k_indices, expert_mask





class SparseMoe(nn.Module):
    def __init__(self, config:Config)->None:
        super().__init__()
        self.top_k = config.top_k
        self.hidden_dim = config.hidden_dim
        self.expert_num = config.expert_num

        self.experts = nn.ModuleList(
            BasicExpert(config.hidden_dim, config.hidden_dim, config.hidden_dim) for _ in range(config.expert_num)
        )

        self.router = Router(config)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        batch_size, seq_len, hidden_dim = x.size()
        hidden_state = x.view(-1, hidden_dim)

        # 路由
        router_logits, router_weight, top_k_incies, expert_mask = self.router(hidden_state)

        # 初始化一个最终输出的state
        # 由于是中途创建的张量, 所以要将它的类型和device设置为hidden_state一致
        output_state = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_state.dtype,
            device=hidden_state.device
        )

        # 遍历每一个expert来计算: 计算出那些选中前top_k个token的state计算出来
        for expert_idx in range(self.expert_num):
            expert_layer = self.experts[expert_idx]
            cur_expert_mask = expert_mask[expert_idx]
            router_weigt_idx, top_x = torch.where(expert_mask[expert_idx])
            cur_state = hidden_state.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim)

            cur_hidden_state = expert_layer(
                cur_state
            ) * router_weight[top_x, router_weigt_idx].unsqueeze(-1)

            output_state.index_add_(
                0,
                top_x,
                cur_hidden_state.to(hidden_state.dtype)
            )
        output_state = output_state.reshape(batch_size, seq_len, hidden_dim)
        return output_state, router_logits
    



        
        


    





