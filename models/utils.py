import torch.nn as nn


class TupleExtractor(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]
