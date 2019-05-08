import torch
from torch import nn
import torch.nn.functional as F

from ethicml.implementations.vfae_modules.decoder import Decoder


class VFAENetwork(nn.Module):
    """
    Implements a generative model with two layers of stochastic variables,
    where both are conditional, i.e.:

    p(x, z1, z2, y | s) = p(z2) p(y) p(z1 | z2, y) p(x | z1, s)

    with q(z1 | x, s) q(z2 | z1, y) q(y | z1) being the variational posteriors.
    """

    def __init__(self, dataset, input_size):
        super(VFAENetwork, self).__init__()
        torch.manual_seed(888)

        l_d = 50

        self.z1_enc_hl = nn.Linear(input_size+1, 100)
        self.z1_enc_bn = nn.BatchNorm1d(100)
        self.z1_enc_mu = nn.Linear(100, l_d)
        self.z1_enc_logvar = nn.Linear(100, l_d)

        self.z2_enc_hl = nn.Linear(l_d+1, 100)
        self.z2_enc_bn = nn.BatchNorm1d(100)
        self.z2_enc_mu = nn.Linear(100, l_d)
        self.z2_enc_logvar = nn.Linear(100, l_d)

        self.z1_dec_hl = nn.Linear(l_d+1, 100)
        self.z1_dec_bn = nn.BatchNorm1d(100)
        self.z1_dec_mu = nn.Linear(100, l_d)
        self.z1_dec_logvar = nn.Linear(100, l_d)

        self.x_dec = Decoder(dataset)

        self.ypred = nn.Linear(l_d, 1)

    def encode_z1(self, x, s):
        z1_h1 = self.z1_enc_bn(F.relu(self.z1_enc_hl(torch.cat((x, s), 1))))
        return self.z1_enc_mu(z1_h1), self.z1_enc_logvar(z1_h1)

    def encode_z2(self, z1, y):
        z2_h1 = self.z2_enc_bn(F.relu(self.z2_enc_hl(torch.cat((z1, y), 1))))
        return self.z2_enc_mu(z2_h1), self.z2_enc_logvar(z2_h1)

    def decode_z1(self, z2, y):
        hl = self.z1_dec_bn(F.relu(self.z1_dec_hl(torch.cat((z2, y), 1))))
        return self.z1_dec_mu(hl), self.z1_dec_logvar(hl)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, s, y):
        z1_mu, z1_logvar = self.encode_z1(x, s)
        # z1 = F.sigmoid(reparameterize(z1_mu, z1_logvar))
        z1 = self.reparameterize(z1_mu, z1_logvar)

        z2_mu, z2_logvar = self.encode_z2(z1, y)
        # z2 = F.sigmoid(reparameterize(z2_mu, z2_logvar))
        z2 = self.reparameterize(z2_mu, z2_logvar)

        z1_dec_mu, z1_dec_logvar = self.decode_z1(z2, y)
        # z1_dec = F.sigmoid(reparameterize(z1_dec_mu, z1_dec_logvar))
        z1_dec = self.reparameterize(z1_dec_mu, z1_dec_logvar)

        x_dec = self.x_dec(z1_dec, s)

        # y_pred_hl = self.yp_hl(z1)
        y_pred = torch.sigmoid(self.ypred(z1))

        z1_triplet = z1, z1_mu, z1_logvar
        z2_triplet = z2, z2_mu, z2_logvar
        z1_d_triplet = z1_dec, z1_dec_mu, z1_dec_logvar

        return z1_triplet, z2_triplet, z1_d_triplet, x_dec, y_pred
