"""
Implementation of Beutel's adversarially learned fair representations
"""
# Disable pylint checking overwritten method signatures. Pytorch forward passes use **kwargs
# pylint: disable=arguments-differ

import random
import argparse
from typing import List, Any
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from ethicml.implementations.utils import load_data_from_flags, save_transformations
from .pytorch_common import CustomDataset


STRING_TO_ACTIVATION_MAP = {
    "Sigmoid()": nn.Sigmoid()
}

STRING_TO_LOSS_MAP = {
    "BCELoss()": nn.BCELoss()
}


def train_and_transform(train, test, flags):
    """Train the fair autoencoder on the training data and then transform both training and test"""
    random.seed(888)
    np.random.seed(888)
    torch.manual_seed(888)
    torch.cuda.manual_seed_all(888)

    train_data = CustomDataset(train)
    size = int(train_data.size)
    s_size = int(train_data.s_size)
    y_size = int(train_data.y_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=flags['batch_size'],
                                               shuffle=False)

    test_data = CustomDataset(test)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=flags['batch_size'],
                                              shuffle=False)

    # convert flags to Python objects
    enc_activation = STRING_TO_ACTIVATION_MAP[flags['enc_activation']]
    adv_activation = STRING_TO_ACTIVATION_MAP[flags['adv_activation']]
    y_loss_fn = STRING_TO_LOSS_MAP[flags['y_loss']]
    s_loss_fn = STRING_TO_LOSS_MAP[flags['s_loss']]

    enc = Encoder(flags['enc_size'], size, enc_activation)
    adv = Adversary(flags['fairness'], flags['adv_size'], flags['enc_size'][-1], s_size,
                    adv_activation)
    pred = Predictor(flags['pred_size'], flags['enc_size'][-1], y_size, adv_activation)
    model = Model(enc, adv, pred)

    optimizer_y = torch.optim.Adam(model.parameters())
    optimizer_s = torch.optim.Adam(model.parameters())

    for i in range(flags['epochs']):
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

                if flags['fairness'] == "Eq. Opp":
                    mask = class_label.ge(0.5)
                elif flags['fairness'] == "Eq. Odds":
                    raise NotImplementedError("Not implemented Eq. Odds yet")
                elif flags['fairness'] == "DI":
                    mask = torch.ones(s_pred.shape).byte()
                s_loss = s_loss_fn(s_pred, torch.masked_select(sens_label, mask).view(-1, s_size))

                optimizer_s.zero_grad()
                s_loss.backward()
                optimizer_s.step()

    train_to_return: List[Any] = []
    test_to_return: List[Any] = []

    for embedding, _, _ in train_loader:
        train_to_return += enc(embedding)[0].data.numpy().tolist()

    for embedding, _, _ in test_loader:
        test_to_return += enc(embedding)[0].data.numpy().tolist()

    return pd.DataFrame(train_to_return), pd.DataFrame(test_to_return)


class GradReverse(Function):
    """Gradient reversal layer"""
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def _grad_reverse(features):
    return GradReverse.apply(features)


class Encoder(nn.Module):
    """Encoder of the GAN"""
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
        grad_reversed_encoded = _grad_reverse(encoded)
        return encoded, grad_reversed_encoded


class Adversary(nn.Module):
    """Adversary of the GAN"""
    def __init__(self, fairness: str, adv_size: List[int], init_size: int, s_size: int,
                 activation: nn.Module):
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
    """Predictor of the GAN"""
    def __init__(self, pred_size: List[int], init_size: int, class_label_size: int,
                 activation: nn.Module):
        super().__init__()
        self.predictor = nn.Sequential()
        if not pred_size:  # In the case that encoder size [] is specified
            self.predictor.add_module("single adversary layer",
                                      nn.Linear(init_size, class_label_size))
            self.predictor.add_module("single layer adversary activation", activation)
        else:
            self.predictor.add_module("adversary layer 0", nn.Linear(init_size, pred_size[0]))
            self.predictor.add_module("adversary activation 0", activation)
            for k in range(len(pred_size) - 1):
                self.predictor.add_module("adversary layer {}".format(k + 1),
                                          nn.Linear(pred_size[k], pred_size[k + 1]))
                self.predictor.add_module("adversary activation {}".format(k + 1), activation)
            self.predictor.add_module("adversary last layer",
                                      nn.Linear(pred_size[-1], class_label_size))
            self.predictor.add_module("adversary last activation", activation)

    def forward(self, x):
        return self.predictor(x)


class Model(nn.Module):
    """Whole GAN model"""
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


def main():
    """Load data from feather files, pass it to `train_and_transform` and then save the result"""
    parser = argparse.ArgumentParser()

    # paths to the files with the data
    parser.add_argument("--train_x", required=True)
    parser.add_argument("--train_s", required=True)
    parser.add_argument("--train_y", required=True)
    parser.add_argument("--test_x", required=True)
    parser.add_argument("--test_s", required=True)
    parser.add_argument("--test_y", required=True)

    # paths to where the processed inputs should be stored
    parser.add_argument("--train_new", required=True)
    parser.add_argument("--test_new", required=True)

    # model parameters
    parser.add_argument("--fairness", required=True)
    parser.add_argument("--enc_size", type=int, nargs='+', required=True)
    parser.add_argument("--adv_size", type=int, nargs='+', required=True)
    parser.add_argument("--pred_size", type=int, nargs='+', required=True)
    parser.add_argument("--enc_activation", required=True)
    parser.add_argument("--adv_activation", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--y_loss", required=True)
    parser.add_argument("--s_loss", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()
    flags = vars(args)  # convert args object to a dictionary

    train, test = load_data_from_flags(flags)
    save_transformations(train_and_transform(train, test, flags),
                         (flags['train_new'], flags['test_new']))


if __name__ == "__main__":
    main()
