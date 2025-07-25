import torch
from torch import nn


class VGG(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        kernel_size: int = 3,
    ):
        super(VGG, self).__init__()
