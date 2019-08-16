import torch
from torch import nn
import torch.distributions as td


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid = nn.Linear(2, 20)
        self.mu = nn.Linear(20, 10)
        self.logvar = nn.Linear(20, 10)
        self.decoder = nn.Linear(11, 10)
        self.decoder_1 = nn.Linear(10, 1)

    def forward(self, z, s):
        z = torch.relu(self.hid(z))
        mu = self.mu(z)
        logvar = self.logvar(z)
        dist = td.Normal(loc=mu, scale=torch.exp(logvar))
        y_pred = torch.relu(self.decoder(torch.cat([dist.sample(), s], dim=1)))
        y_pred = self.decoder_1(y_pred)
        return dist, torch.sigmoid(y_pred)
