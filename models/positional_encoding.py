import math

import torch
from torch import Tensor
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 50000,
        div_factor: float = 10000.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(div_factor) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ContinuousPositionalEncoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        shortest_period: float = 0.1,
        longest_period: float = 300.0,
    ):
        """
        Create a continuous positional encoding for a sequence.

        :param d_model: Dimension of the model.
        :param dropout: Dropout rate.
        :param shortest_period: Period of the shortest sinusoid (controls the shortest amount of
            time that the model can distinguish).
        :param longest_period: Period of the longest sinusoid (controls the longest amount of time
            that the model can distinguish).
        """
        super().__init__()
        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        # Compute the div_term so that the sinusoids have the correct period (between
        # smallest_period_duration and largest_period_duration).
        highest_frequency = 1 / shortest_period
        div_factor = longest_period * highest_frequency
        div_term = (
            math.tau
            * highest_frequency
            * (torch.exp(torch.arange(0, d_model, 2) * (-math.log(div_factor) / (d_model - 2))))
        )

        # Add a dimension for the batch and the sequence length
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        # print(f"{div_term.shape = }\n{div_term}\n")
        self.register_buffer("div_term", div_term)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        :param x: Tensor of shape ``[batch_size, seq_len, d_model]``
        :param positions: Tensor of shape ``[batch_size, seq_len]`` containing the positions of
            each element in each sequence of the batch.
        """
        # print(f"{x.shape = }\n{x}\n")
        positions = positions.unsqueeze(-1)
        # print(f"{positions.shape = }\n{positions}\n")
        pe = torch.zeros_like(x)
        scaled_position = positions * self.div_term
        pe[:, :, 0::2] = torch.sin(scaled_position)
        pe[:, :, 1::2] = torch.cos(scaled_position[..., : self.d_model // 2])
        # ^ Division needed for when d_model is odd ^
        # print(f"{pe.shape = }\n{pe}\n")
        x = x + pe
        return self.dropout(x)


class TreePositionalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_depth: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth
        self.linear = nn.Linear(2 * max_depth, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        :param x: Tensor of shape ``[batch_size, seq_len, d_model]``
        :param positions: List containing for each element of x the path in the tree leading to
            the corresponding node from the root node in the form [i, n] with i the child index and
            n the number of children of the parent node.
        """
        positions_projected_into_model_shape = self.linear(positions)
        return self.dropout(x + positions_projected_into_model_shape)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d_model = 22
    seq_len = 38.4
    # div_factor = 100
    #
    # pe = PositionalEncoder(d_model, 0, max_len=50000, div_factor=div_factor)
    # y = pe(torch.zeros(1, seq_len, d_model))
    # print(y.shape)

    # plt.figure(figsize=(15, 5))
    # for i in range(seq_len):
    #     plt.plot(y[0, i, :].unsqueeze(-1).detach().numpy(), label=f"Position {i}")
    # plt.title("Positional Encoding")
    # plt.xlabel("Position")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.show(block=False)

    # Plot matrix
    # plt.figure(figsize=(15, 5))
    # plt.imshow(y[0, :, :], cmap="viridis", aspect="auto")
    # plt.colorbar()
    # plt.title("Positional Encoding")
    # plt.xlabel("Dimension")
    # plt.ylabel("Position")
    # plt.show(block=False)

    smallest_period_duration = 0.1
    largest_period_duration = 300
    pe = ContinuousPositionalEncoder(
        d_model,
        0,
        shortest_period=smallest_period_duration,
        longest_period=largest_period_duration,
    )
    res = 1000
    positions = torch.arange(0, seq_len, 1 / res)
    print(positions)
    y = pe(torch.zeros(1, int(seq_len * res), d_model), positions)
    print(y.shape)

    # Plot matrix
    plt.figure(figsize=(15, 5))
    plt.imshow(
        y[0, :, :],
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
        extent=[-0.5, d_model - 0.5, seq_len, 0],
    )
    plt.colorbar()
    plt.title(
        f"Continuous Positional Encoding\n{d_model = }\n{smallest_period_duration = }\n{largest_period_duration = }"
    )
    plt.xticks(range(d_model))
    plt.xlabel("Dimension")
    plt.ylabel("Time")
    plt.subplots_adjust(top=0.82)
    plt.show()
