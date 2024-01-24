import torch.nn as nn


class TupleExtractor(nn.Module):
    """
    This simple module extracts a component of a tuple returned by the module it follows.
    """

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]
