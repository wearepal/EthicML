from torch import nn


class Encoder(nn.Module):
    def __init__(self, enc_size, init_size, ld, activation=nn.ReLU()):
        super().__init__()
        self.encoder = nn.Sequential()
        if not enc_size:  # In the case that encoder size [] is specified
            self.z1_enc_mu = nn.Linear(init_size, ld)
            self.z1_enc_logvar = nn.Linear(init_size, ld)
        else:
            self.encoder.add_module("encoder layer 0", nn.Linear(init_size, enc_size[0]))
            if activation:
                self.encoder.add_module("encoder activation 0", activation)
            self.encoder.add_module("batch norm 0", nn.BatchNorm1d(enc_size[0]))
            for k in range(len(enc_size) - 1):
                self.encoder.add_module(
                    f"encoder layer {k+1}", nn.Linear(enc_size[k], enc_size[k + 1])
                )
                if activation:
                    self.encoder.add_module(f"encoder activation {k+1}", activation)
                self.encoder.add_module(
                    f"encoder batch norm {k+1}", nn.BatchNorm1d(enc_size[k + 1])
                )
            self.z1_enc_mu = nn.Linear(enc_size[-1], ld)
            self.z1_enc_logvar = nn.Linear(enc_size[-1], ld)

    def forward(self, input):
        step = self.encoder(input)
        return self.z1_enc_mu(step), self.z1_enc_logvar(step)
