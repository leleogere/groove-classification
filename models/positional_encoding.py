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


# class ContinuousPositionalEmbedding(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         tensor_to_position: Callable[[Tensor], Tensor],
#         dropout: float = 0.1,
#         importance_factor: float = 0.1,
#     ):
#         """
#         Create a continuous positional encoding for a sequence.
#
#         :param d_model: Dimension of the model.
#         :param tensor_to_position: Function that takes a tensor and returns and return the position
#             of each element in the tensor. For a tensor of shape ``[batch_size, seq_len, d_model]``,
#             the output should have shape ``[batch_size, seq_len]`` (one position for each element).
#         :param dropout: Dropout rate.
#         :param importance_factor: Factor controlling the scale of the positional encoding.
#         """
#         super().__init__()
#         self.d_model = d_model
#         self.tensor_to_position = tensor_to_position
#
#         self._verify_tensor_to_position()
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         div_term = (
#             (
#                 torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#                 * importance_factor
#             )
#             .unsqueeze(0)
#             .unsqueeze(0)
#         )
#         # print(f"{div_term.shape = }\n{div_term}\n")
#         self.register_buffer("div_term", div_term)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         :param x: Tensor of shape ``[batch_size, seq_len, d_model]``
#         """
#         print(f"{x.shape = }\n{x}\n")
#         position = self.tensor_to_position(x).unsqueeze(-1)
#         print(f"{position.shape = }\n{position}\n")
#         pe = torch.zeros_like(x)
#         scaled_position = position * self.div_term
#         pe[:, :, 0::2] = torch.sin(scaled_position)
#         pe[:, :, 1::2] = torch.cos(scaled_position[..., : self.d_model // 2])
#         # ^ Division needed for when d_model is odd ^
#         print(f"{pe.shape = }\n{pe}\n")
#         x = x + pe
#         return self.dropout(x)
#
#     def _verify_tensor_to_position(self):
#         batch = 4
#         seq_len = 50
#         x = torch.ones((batch, seq_len, self.d_model))
#         pos = self.tensor_to_position(x)
#         if not (pos.shape == (batch, seq_len)):
#             raise ValueError(f"Expected shape {(batch, seq_len)}, got {pos.shape}.")


# class ContinuousPositionalEmbedding(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         tensor_to_position: Callable[[Tensor], Tensor],
#         dropout: float = 0.1,
#         importance_factor: float = 0.1,
#     ):
#         """
#         Create a continuous positional encoding for a sequence.
#
#         :param d_model: Dimension of the model.
#         :param tensor_to_position: Function that takes a tensor and returns and return the position
#             of each element in the tensor.
#         :param dropout: Dropout rate.
#         :param importance_factor: Factor controlling the scale of the positional encoding.
#         """
#         super().__init__()
#         self.d_model = d_model
#         self.tensor_to_position = tensor_to_position
#
#         self._verify_tensor_to_position()
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         div_term = (
#             torch.exp(torch.arange(0, d_model) * (-math.log(10000.0) / d_model)) * importance_factor
#         )
#         # print(f"{div_term.shape = }\n{div_term}\n")
#         self.register_buffer("div_term", div_term)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         :param x: Tensor of shape ``[batch_size, seq_len, d_model]``
#         """
#         # print(f"{x.shape = }\n{x}\n")
#         pos = self.tensor_to_position(x)
#         # print(f"{pos.shape = }\n{pos}\n")
#         pos_embedding = torch.sin(pos.unsqueeze(-1) * torch.exp(self.div_term))
#         # print(f"{pos_embedding.shape = }\n{pos_embedding}\n")
#         x = x + pos_embedding
#         return self.dropout(x)
#
#     def _verify_tensor_to_position(self):
#         batch = 4
#         seq_len = 50
#         x = torch.ones((batch, seq_len, self.d_model))
#         pos = self.tensor_to_position(x)
#         if not (pos.shape == (batch, seq_len)):
#             raise ValueError(f"Expected shape {(batch, seq_len)}, got {pos.shape}.")


# if __name__ == "__main__":
#     x = torch.tensor([
#         [
#             [10, 10, 0.0],
#             [10, 10, 0.1],
#             [10, 10, 0.3],
#             [10, 10, 0.3],
#         ],
#         [
#             [10, 10, 0.1],
#             [10, 10, 0.0],
#             [10, 10, 0.5],
#             [10, 10, 0.3],
#         ],
#         [
#             [10, 10, 0.0],
#             [10, 10, 0.3],
#             [10, 10, 0.0],
#             [10, 10, 0.3],
#         ],
#         [
#             [10, 10, 0.2],
#             [10, 10, 0.0],
#             [10, 10, 0.0],
#             [10, 10, 0.3],
#         ],
#         [
#             [10, 10, 0.1],
#             [10, 10, 0.0],
#             [10, 10, 0.2],
#             [10, 10, 0.0],
#         ],
#     ])
#
#     def tensor_to_position(x):
#         return torch.cumsum(x[:, :, -1], dim=-1)
#
#     # print(x)
#     # print(x.shape)
#     # print(x[:, :, -1])
#     # print(tensor_to_position(x))
#     # print(tensor_to_position(x).shape)
#
#     time_embedding = ContinuousPositionalEmbedding(
#         d_model=3,
#         tensor_to_position=tensor_to_position,
#     )
#     print(time_embedding(x))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d_model = 24
    seq_len = 30
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

    smallest_period_duration = 0.5
    largest_period_duration = 600
    pe = ContinuousPositionalEncoder(
        d_model,
        0,
        shortest_period=smallest_period_duration,
        longest_period=largest_period_duration,
    )
    res = 1000
    positions = torch.arange(0, seq_len, 1 / res)
    print(positions)
    y = pe(torch.zeros(1, seq_len * res, d_model), positions)
    print(y.shape)

    # Plot matrix
    plt.figure(figsize=(15, 5))
    plt.imshow(
        y[0, :, :],
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
        extent=[0, d_model, seq_len, 0],
    )
    plt.colorbar()
    plt.title(
        f"Continuous Positional Encoding\n{smallest_period_duration = }\n{largest_period_duration = }"
    )
    plt.xlabel("Dimension")
    plt.ylabel("Time")
    plt.show()
