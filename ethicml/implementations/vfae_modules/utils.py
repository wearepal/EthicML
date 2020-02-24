"""Implementation for Louizos et al Variational Fair Autoencoder."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..pytorch_common import quadratic_time_mmd


def kullback_leibler(
    mu1: Tensor, logvar1: Tensor, mu2: Optional[Tensor] = None, logvar2: Optional[Tensor] = None
) -> Tensor:
    """KL Divergence.

    Args:
        mu1:
        logvar1:
        mu2:
        logvar2:

    Returns:
        Tensorof divergence in each dim.
    """
    mu2 = mu2 if mu2 is not None else torch.tensor([0.0])
    logvar2 = logvar2 if logvar2 is not None else torch.tensor([0.0])
    # return -0.5 * torch.sum(1 + (logvar1-logvar2) - (mu1-mu2).pow(2) - (logvar1-logvar2).exp())
    return (
        0.5
        * (logvar2 - logvar1 - 1 + ((logvar1).exp() + ((mu1 - mu2) ** 2)) / (logvar2).exp()).sum()
    )


def loss_function(flags, z1_triplet, z2_triplet, z1_d_triplet, data_triplet, x_dec, y_pred):
    """Loss function for VFAE.

    Args:
        flags:
        z1_triplet:
        z2_triplet:
        z1_d_triplet:
        data_triplet:
        x_dec:
        y_pred:

    Returns:
        Tuple of prediction loss, reconstruction loss, KL Divergence and MMD.
    """
    z1, z1_mu, z1_logvar = z1_triplet
    if flags.supervised:
        z2, z2_mu, z2_logvar = z2_triplet
        z1_dec, z1_dec_mu, z1_dec_logvar = z1_d_triplet
    x, s, y = data_triplet

    reconstruction_loss = F.mse_loss(x_dec, x, reduction="sum")

    if flags.fairness == "DI":
        z1_s0 = torch.masked_select(z1, s.le(0.5)).view(-1, 50)
        z1_s1 = torch.masked_select(z1, s.ge(0.5)).view(-1, 50)
    elif flags.fairness == "Eq. Opp":
        z1_s0 = torch.masked_select(z1, s.le(0.5)).view(-1, 50)
        y_s0 = torch.masked_select(y, s.le(0.5)).view(-1, 1)
        z1_s0_y1 = torch.masked_select(z1_s0, y_s0.ge(0.5)).view(-1, 50)
        z1_s1 = torch.masked_select(z1, s.ge(0.5)).view(-1, 50)
        y_s1 = torch.masked_select(y, s.ge(0.5)).view(-1, 1)
        z1_s1_y1 = torch.masked_select(z1_s1, y_s1.ge(0.5)).view(-1, 50)

        z1_s0 = z1_s0_y1
        z1_s1 = z1_s1_y1
    else:
        raise NotImplementedError("Only DI and Eq.Opp implementesd so far")

    mmd_loss = quadratic_time_mmd(z1_s0, z1_s1, 2.5)

    if flags.supervised:
        first_kl = kullback_leibler(z2_mu, z2_logvar)
        second_kl = kullback_leibler(z1_dec_mu, z1_dec_logvar, z1_mu, z1_logvar)
        # second_kl = F.kl_div(z1_dec, z1, reduction='sum')
        # second_kl = (z1_dec.sum()+1e-10).log() - (z1.sum()+1e-10).log()
        kl_div = first_kl + second_kl
        prediction_loss = F.binary_cross_entropy(y_pred, y, reduction="sum")
    else:
        kl_div = torch.zeros(1)
        prediction_loss = torch.zeros(1)

    return prediction_loss, reconstruction_loss, kl_div, 100 * mmd_loss
