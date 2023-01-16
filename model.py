import torch
from torch import nn


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        mid_activ: nn.Module = nn.GELU(),
        last_activ: nn.Module = nn.LeakyReLU(),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            mid_activ,
            nn.Linear(hidden_size, output_size),
            last_activ,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activ: nn.Module = nn.LeakyReLU(),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size, stride, padding),
            activ,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
