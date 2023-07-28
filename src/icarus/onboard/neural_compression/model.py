import torch
from torch import nn
from torch.nn import ReLU


class ResBlock(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.d1 = nn.Linear(dim, dim)
        self.d2 = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, x):
        skip = x

        x = self.activation(self.d1(x))
        x = self.d2(x)

        x += skip
        x = self.activation(x)
        return x


class ResModel(nn.Module):
    def __init__(self, in_coords, out_values, dim, n_blocks):
        super().__init__()
        self.activation = ReLU()

        self.posenc = PositionalEncoding(8, 20)
        self.d_in = nn.Linear(in_coords * 20 * 2, dim)

        meta_blocks = [nn.Linear(dim, dim) for _ in range(n_blocks)]
        self.meta_blocks = nn.ModuleList(meta_blocks)

        head_blocks = [nn.Linear(dim, dim) for _ in range(n_blocks // 2)]
        self.head_blocks = nn.ModuleList(head_blocks)

        self.d_out = nn.Linear(dim, out_values)

    def forward(self, x):
        x = self.posenc(x)
        x = self.activation(self.d_in(x))
        for b in self.meta_blocks:
            x = self.activation(b(x))
        for b in self.head_blocks:
            x = self.activation(b(x))
        x = self.d_out(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs)  # (num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (batch, in_features)
        Outputs:
            out: (batch, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2) * self.freqs.unsqueeze(
            dim=-1
        )  # (batch, num_freqs, in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)  # (batch, num_freqs*in_features)
        out = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # (batch, 2*num_freqs*in_features)
        return out
