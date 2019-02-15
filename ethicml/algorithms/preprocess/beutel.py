"""
Implementation of Beute's adversarially learned fair representations
"""

# Disable pylint checking overwritten method signatures. Pytorch forward passes use **kwargs
# pylint: disable=arguments-differ

import random
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from ..dataloader_funcs import CustomDataset
from .pre_algorithm import PreAlgorithm


class Beutel(PreAlgorithm):
    def __init__(self,
                 fairness: str = "DI",
                 enc_size: List[int] = None,
                 adv_size: List[int] = None,
                 pred_size: List[int] = None,
                 enc_activation=nn.Sigmoid(),
                 adv_activation=nn.Sigmoid(),
                 batch_size: int = 64,
                 y_loss=nn.BCELoss(),
                 s_loss=nn.BCELoss(),
                 epochs=50):
        # pylint: disable=too-many-arguments
        self.fairness = fairness
        self.enc_size: List[int] = [40] if enc_size is None else enc_size
        self.adv_size: List[int] = [40] if adv_size is None else adv_size
        self.pred_size: List[int] = [40] if pred_size is None else pred_size
        self.enc_activation = enc_activation
        self.adv_activation = adv_activation
        self.batch_size = batch_size
        self.y_loss = y_loss
        self.s_loss = s_loss
        self.epochs = epochs
        random.seed(888)
        np.random.seed(888)
        torch.manual_seed(888)
        torch.cuda.manual_seed_all(888)

    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # pylint: disable=too-many-statements

        train_data = CustomDataset(train)
        size = int(train_data.size)
        s_size = int(train_data.s_size)
        y_size = int(train_data.y_size)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=False)

        test_data = CustomDataset(test)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False)

        class GradReverse(Function):
            @staticmethod
            def forward(ctx, x):
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg()

        def grad_reverse(features):
            return GradReverse.apply(features)

        class Encoder(nn.Module):
            def __init__(self, enc_size: List[int], init_size: int, activation):
                super().__init__()
                self.encoder = nn.Sequential()
                if not enc_size:  # In the case that encoder size [] is specified
                    self.encoder.add_module("single encoder layer", nn.Linear(init_size, init_size))
                    self.encoder.add_module("single layer encoder activation", activation)
                else:
                    self.encoder.add_module("encoder layer 0", nn.Linear(init_size, enc_size[0]))
                    self.encoder.add_module("encoder activation 0", activation)
                    for k in range(len(enc_size) - 1):
                        self.encoder.add_module("encoder layer {}".format(k + 1),
                                                nn.Linear(enc_size[k], enc_size[k + 1]))
                        self.encoder.add_module("encoder activation {}".format(k + 1), activation)

            def forward(self, x):
                encoded = self.encoder(x)
                grad_reversed_encoded = grad_reverse(encoded)
                return encoded, grad_reversed_encoded

        class Adversary(nn.Module):
            def __init__(self, fairness: str, adv_size: List[int], init_size: int, s_size: int, activation: nn.Module):
                super().__init__()
                self.fairness = fairness
                self.init_size = init_size
                self.adversary = nn.Sequential()
                if not adv_size:  # In the case that encoder size [] is specified
                    self.adversary.add_module("single adversary layer", nn.Linear(init_size, s_size))
                    self.adversary.add_module("single layer adversary activation", activation)
                else:
                    self.adversary.add_module("adversary layer 0", nn.Linear(init_size, adv_size[0]))
                    self.adversary.add_module("adversary activation 0", activation)
                    for k in range(len(adv_size) - 1):
                        self.adversary.add_module("adversary layer {}".format(k + 1),
                                                  nn.Linear(adv_size[k], adv_size[k + 1]))
                        self.adversary.add_module("adversary activation {}".format(k + 1), activation)
                    self.adversary.add_module("adversary last layer", nn.Linear(adv_size[-1], s_size))
                    self.adversary.add_module("adversary last activation", activation)

            def forward(self, x, y):
                if self.fairness == "Eq. Opp":
                    mask = y.ge(0.5)
                    x = torch.masked_select(x, mask).view(-1, self.init_size)
                    x = self.adversary(x)
                elif self.fairness == "Eq. Odds":
                    raise NotImplementedError("Not implemented equalized odds yet")
                elif self.fairness == "DI":
                    x = self.adversary(x)
                return x

        class Predictor(nn.Module):
            def __init__(self, pred_size: List[int], init_size: int, class_label_size: int, activation: nn.Module):
                super().__init__()
                self.predictor = nn.Sequential()
                if not pred_size:  # In the case that encoder size [] is specified
                    self.predictor.add_module("single adversary layer", nn.Linear(init_size, class_label_size))
                    self.predictor.add_module("single layer adversary activation", activation)
                else:
                    self.predictor.add_module("adversary layer 0", nn.Linear(init_size, pred_size[0]))
                    self.predictor.add_module("adversary activation 0", activation)
                    for k in range(len(pred_size) - 1):
                        self.predictor.add_module("adversary layer {}".format(k + 1),
                                                  nn.Linear(pred_size[k], pred_size[k + 1]))
                        self.predictor.add_module("adversary activation {}".format(k + 1), activation)
                    self.predictor.add_module("adversary last layer", nn.Linear(pred_size[-1], class_label_size))
                    self.predictor.add_module("adversary last activation", activation)

            def forward(self, x):
                return self.predictor(x)

        class Model(nn.Module):
            def __init__(self, enc, adv, pred):
                super().__init__()
                self.enc = enc
                self.adv = adv
                self.pred = pred

            def forward(self, x, y):
                encoded = self.enc(x)
                s_hat = self.adv(encoded[1], y)
                y_hat = self.pred(encoded[0])
                return encoded, s_hat, y_hat

        enc = Encoder(self.enc_size, size, self.enc_activation)
        adv = Adversary(self.fairness, self.adv_size, self.enc_size[-1], s_size, self.adv_activation)
        pred = Predictor(self.pred_size, self.adv_size[-1], y_size, self.adv_activation)
        model = Model(enc, adv, pred)

        y_loss_fn = self.y_loss
        s_loss_fn = self.s_loss

        optimizer_y = torch.optim.Adam(model.parameters())
        optimizer_s = torch.optim.Adam(model.parameters())

        for i in range(self.epochs):
            if i % 4 == 0:
                for embedding, sens_label, class_label in train_loader:
                    _, s_pred, y_pred = model(embedding, class_label)

                    y_loss = y_loss_fn(y_pred, class_label)

                    optimizer_y.zero_grad()
                    y_loss.backward()
                    optimizer_y.step()
            else:
                for embedding, sens_label, class_label in train_loader:
                    _, s_pred, y_pred = model(embedding, class_label)

                    if self.fairness == "Eq. Opp":
                        mask = class_label.ge(0.5)
                    elif self.fairness == "Eq. Odds":
                        raise NotImplementedError("Not implemented Eq. Odds yet")
                    elif self.fairness == "DI":
                        mask = torch.ones(s_pred.shape).byte()
                    s_loss = s_loss_fn(s_pred, torch.masked_select(sens_label, mask).view(-1, s_size))

                    optimizer_s.zero_grad()
                    s_loss.backward()
                    optimizer_s.step()

        train = []
        test = []

        for embedding, _, _ in train_loader:
            train += enc(embedding)[0].data.numpy().tolist()

        for embedding, _, _ in test_loader:
            test += enc(embedding)[0].data.numpy().tolist()

        return pd.DataFrame(train), pd.DataFrame(test)

    @property
    def name(self) -> str:
        return "Beutel"
