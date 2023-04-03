import torch.nn as nn
import torch

class UAP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.uap = nn.Parameter(torch.zeros(size = (1, 3200), requires_grad=True))

    def forward(self):
        return self.uap