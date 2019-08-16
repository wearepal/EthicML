from typing import List

import torch
from torch import nn
import torch.distributions as td

from ethicml.data import Dataset
from ethicml.implementations.pytorch_common import CustomDataset

HID_SIZE = 100
L_D = 50


class Features(nn.Module):
    def __init__(self, data: CustomDataset, dataset: Dataset):
        super().__init__()
        self.data = data
        self.dataset = dataset
        self.in_feats = data.size
        self.hid = nn.Linear(self.in_feats, 20)
        self.mu = nn.Linear(20, 4)
        self.logvar = nn.Linear(20, 4)
        self.decoder = nn.Linear(5, 20)
        self.decoder_1 = nn.Linear(20, 2)
        self.decoder_2 = nn.Linear(20, 2)

    def forward(self, x, s):
        x = torch.relu(self.hid(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        dist = td.Normal(loc=mu, scale=torch.exp(logvar))

        dec = torch.relu(self.decoder(torch.cat([dist.sample(), s], dim=1)))
        dec = td.Normal(self.decoder_1(dec), self.decoder_2(dec))

        return dist, dec
