"""Implementation for Louizos et al Variational Fair Autoencoder."""
# pylint: disable=arguments-differ

from typing import Any, List, NamedTuple, Optional, Tuple

import torch
from torch import Tensor, nn

from ethicml.data.dataset import CSVDataset

from .decoder import Decoder
from .encoder import Encoder


class LvInfo(NamedTuple):
    """Hold Latent Variable Information."""

    embed: Tensor
    mean: Tensor
    logvar: Tensor


class VFAENetwork(nn.Module):
    """Implements a generative model with two layers of stochastic variables.

    Where both are conditional, i.e.:

    p(x, z1, z2, y | s) = p(z2) p(y) p(z1 | z2, y) p(x | z1, s)

    with q(z1 | x, s) q(z2 | z1, y) q(y | z1) being the variational posteriors.
    """

    def __init__(
        self,
        dataset: CSVDataset,
        supervised: bool,
        input_size: int,
        latent_dims: int,
        z1_enc_size: List[int],
        z2_enc_size: List[int],
        z1_dec_size: List[int],
    ):
        super().__init__()
        torch.manual_seed(888)

        self.supervised = supervised

        self.z1_encoder = Encoder(z1_enc_size, input_size + 1, latent_dims)
        if self.supervised:
            self.z2_encoder = Encoder(z2_enc_size, latent_dims + 1, latent_dims)
            self.z1_decoder = Encoder(z1_dec_size, latent_dims + 1, latent_dims)
        self.x_dec = Decoder(dataset)
        self.ypred = nn.Linear(latent_dims, 1)

    def encode_z1(self, x: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode Z1.

        :param x:
        :param s:
        """
        return self.z1_encoder(torch.cat((x, s.view(-1, 1)), 1))

    def encode_z2(self, z1: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode Z2.

        :param z1:
        :param y:
        """
        return self.z2_encoder(torch.cat((z1, y.view(-1, 1)), 1))

    def decode_z1(self, z2: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode Z1.

        :param z2:
        :param y:
        """
        return self.z1_decoder(torch.cat((z2, y.view(-1, 1)), 1))

    @staticmethod
    def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparametrization trick - Leaving as a method to try and control reproducability.

        :param mean:
        :param logvar:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(
        self, x: Tensor, s: Tensor, y: Tensor
    ) -> Tuple[LvInfo, Optional[LvInfo], Optional[LvInfo], Tensor, Optional[Tensor],]:
        """Forward pass for network.

        :param x:
        :param s:
        :param y:
        """
        z1_mu, z1_logvar = self.encode_z1(x, s)
        z1 = self.reparameterize(z1_mu, z1_logvar)

        z2_triplet: Optional[Tuple[Any, Any, Any]]
        z1_d_triplet: Optional[Tuple[Any, Any, Any]]
        y_pred: Optional[torch.Tensor]
        if self.supervised:
            z2_mu, z2_logvar = self.encode_z2(z1, y)
            z2 = self.reparameterize(z2_mu, z2_logvar)

            z1_dec_mu, z1_dec_logvar = self.decode_z1(z2, y)
            z1_dec = self.reparameterize(z1_dec_mu, z1_dec_logvar)

            x_dec = self.x_dec(z1_dec, s)

            y_pred = torch.sigmoid(self.ypred(z1))

            z2_triplet = LvInfo(embed=z2, mean=z2_mu, logvar=z2_logvar)
            z1_d_triplet = LvInfo(embed=z1_dec, mean=z1_dec_mu, logvar=z1_dec_logvar)

        else:
            x_dec = self.x_dec(z1, s)
            z2_triplet = None
            z1_d_triplet = None
            y_pred = None

        z1_triplet = LvInfo(embed=z1, mean=z1_mu, logvar=z1_logvar)

        return z1_triplet, z2_triplet, z1_d_triplet, x_dec, y_pred
