from itertools import groupby

import torch
from torch import nn
from typing import List

from ethicml.data.dataset import Dataset
from ethicml.implementations.vfae_modules.categorical import Categorical


class Decoder(nn.Module):
    def __init__(self, dataset: Dataset, deploy=False):
        super().__init__()
        self._deploy = deploy
        self.features: List[str] = dataset.feature_split['x']

        L_D = 50
        HID_SIZE = 100

        self.shared_net = nn.Sequential()
        in_features = L_D + 1
        # add hidden layers according to the number of units specified in "hidden_sizes"
        for depth, num_units in enumerate([HID_SIZE]):
            self.shared_net.add_module("hidden_layer_%d" % depth, nn.Linear(in_features, num_units))
            self.shared_net.add_module("ReLu %d" % depth, nn.ReLU())
            in_features = num_units  # update input size to next layer

        def _add_output_layer(feature_group: List) -> nn.Sequential:
            n_dims = len(feature_group)
            categorical = n_dims > 1  # feature is categorical if it has more than 1 possible output

            if categorical:
                layer = Categorical(in_features, n_dims)
            else:
                layer = nn.Sequential(nn.Linear(in_features, n_dims))  # , nn.Sigmoid())

            return layer

        self.grouped_features = [
            list(group) for key, group in groupby(self.features, lambda x: x.split('_')[0])
        ]
        self.output_layers = nn.ModuleList(
            [_add_output_layer(feature) for feature in self.grouped_features]
        )

    def forward(self, x, s):
        batch_size = x.size(0)
        decoded = self.shared_net(torch.cat((x, s), 1))
        decoded = torch.cat(
            [layer(decoded).view(batch_size, -1) for layer in self.output_layers], dim=1
        )
        return decoded
