import torch
import torch.nn as nn 

class BasicMoe(nn.Module):
    def __init__(self, feat_in:int, hidden_dim:int, feat_out:int)->None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_out)
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.net(x)
    



