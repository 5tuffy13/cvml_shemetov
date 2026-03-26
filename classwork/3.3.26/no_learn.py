
# NB Проблема нарушения непрерывности массива
import torch
from torch import nn

class SquareModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x ** 2
model = SquareModel()
x = torch.tensor([3.0])
output = model(x)
print(output)