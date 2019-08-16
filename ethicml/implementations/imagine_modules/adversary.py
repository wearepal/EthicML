from torch import nn
from torch.autograd import Function
import torch


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(features):
    return GradReverse.apply(features)


class FeatAdversary(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred = nn.Linear(4, 10)
        self.pred_1 = nn.Linear(10, 1)

    def forward(self, z):
        s = torch.relu(self.pred(grad_reverse(z)))
        s = self.pred_1(s)
        return torch.sigmoid(s)


class PredAdversary(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred = nn.Linear(10, 10)
        self.pred_1 = nn.Linear(10, 1)

    def forward(self, z):
        s = torch.relu(self.pred(grad_reverse(z)))
        s = self.pred_1(s)
        return torch.sigmoid(s)
