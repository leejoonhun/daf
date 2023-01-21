from typing import Tuple, Union

import torch
from torch import nn


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Union[int, Tuple[int, int]],
        activ: nn.Module = nn.LeakyReLU(),
        is_cls: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, tuple):
            modules = [
                nn.Linear(input_dim, hidden_dim[0]),
                nn.Linear(hidden_dim[0], hidden_dim[-1]),
                nn.Linear(hidden_dim[-1], output_dim),
                activ,
            ]
        else:
            modules = [
                nn.Linear(input_dim, hidden_dim),
                activ,
                nn.Linear(hidden_dim, output_dim),
                activ,
            ]

        if is_cls:
            modules.append(nn.Softmax(dim=-1))

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ConvLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]],
        activ: nn.Module = nn.LeakyReLU(),
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, tuple):
            self.net = nn.Sequential(
                nn.Conv1d(
                    input_dim,
                    hidden_dim,
                    kernel_size[0],
                    padding=kernel_size[0] // 2,
                ),
                nn.Conv1d(
                    hidden_dim,
                    output_dim,
                    kernel_size[1],
                    padding=kernel_size[1] // 2,
                ),
                activ,
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(
                    input_dim,
                    output_dim,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                activ,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
