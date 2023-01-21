import math
from typing import Tuple, Union

import torch
from torch import nn

from .base import ConvLayer, MLPLayer


class SequenceGenerator(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        pred_len: int,
        attn_module: nn.Module,
        hidden_dim: Union[int, Tuple[int, int]],
        kernel_size: Union[int, Tuple[int, int]],
    ) -> None:
        super().__init__()
        self.pred_len = pred_len

        self.attn = attn_module
        self.enc = Encoder(feat_dim, hidden_dim, kernel_size)
        self.dec = Decoder(feat_dim, hidden_dim)

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sequence Generator

        Args:
            data (torch.Tensor): input data with shape `(B, D, T)`, where `D = 1 + d`.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - reconstructed & predicted results with shape `(B, D, T + t)`.
                - queries and keys with shape `(B, D, T)`.
        """
        for _ in range(self.pred_len):
            pattern, value = self.enc(data)
            rep, (query, key) = self.attn(pattern, value)
            pred = self.dec(rep)
            data = torch.cat([data, pred[..., -1:]], dim=-1)
        return pred, (query, key)

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)[..., -self.pred_len :]


class SharedAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        self.mlp_q = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )
        self.mlp_k = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )
        self.mlp_o = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )

    def forward(
        self, pattern: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Shared Attention

        Args:
            pattern (torch.Tensor): pattern embeddings with shape `(B, M, T)`.
            value (torch.Tensor): values with shape `(B, D, T)`.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - representations with shape `(B, D, T + 1)`.
                - queries and keys with shape `(B, D, T)`.
        """
        query, key = self.mlp_q(pattern), self.mlp_k(pattern)
        # interpolation mode
        rep_in = self.mlp_o(self.calc_attn(query, key, value))
        # extrapolation mode
        rep_ex = self.mlp_o(
            self.calc_attn(query, key, value, kernel_size=self.kernel_size)
        )
        return torch.cat([rep_in, rep_ex], dim=-1), (query, key)

    def calc_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Calculate attention value

        Args:
            query (torch.Tensor): queries with shape `(B, D, T)`.
            key (torch.Tensor): keys with shape `(B, D, T)`.
            value (torch.Tensor): values with shape `(B, D, T)`.
            kernel_size (Union[int, Tuple[int, int]], optional): kernel size used in encoder,
                required in extrapolation mode.

        Returns:
            torch.Tensor: attention values with shape `(B, D, T + 1)`.
        """
        if isinstance(kernel_size, tuple):
            kernel_size = max(kernel_size)
        if kernel_size:
            unit_len = (kernel_size - 1) // 2
            query = query[..., -1 - unit_len : -1 - unit_len + 1]
            # ? why k, v shape = [..., 96]..?
            key = key[..., kernel_size - unit_len - 1 : -1 - unit_len - 1]
            value = value[..., kernel_size:-1]
        attn_logits = torch.exp(
            query.transpose(1, 2) @ key
            - (query.transpose(1, 2) @ key)
            * torch.eye(query.shape[-1], device=query.device)
            / math.sqrt(query.shape[-1])
        )
        attn_scores = torch.softmax(attn_logits, dim=-1)
        attn_values = (attn_scores @ value.transpose(1, 2)).transpose(1, 2)
        return attn_values


class Encoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: Union[int, Tuple[int, int]],
        kernel_size: Union[int, Tuple[int, int]],
    ) -> None:
        super().__init__()
        self.mlp_v = MLPLayer(
            input_dim=feat_dim,
            output_dim=feat_dim - 1 if feat_dim > 1 else feat_dim,
            hidden_dim=hidden_dim,
        )
        self.conv_p = ConvLayer(
            input_dim=feat_dim,
            output_dim=feat_dim - 1 if feat_dim > 1 else feat_dim,
            hidden_dim=hidden_dim[0] if isinstance(hidden_dim, tuple) else hidden_dim,
            kernel_size=kernel_size,
        )

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Private Encoder

        Args:
            data (torch.Tensor): input data with shape `(B, D, T)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - values with shape `(B, D, T)`.
                - pattern embeddings with shape `(B, M, T)`.
        """
        return self.conv_p(data), self.mlp_v(data)


class Decoder(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp_z = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """Private Decoder

        Args:
            rep (torch.Tensor): representations with shape `(B, D, T + i)`, where `i in range(1, t + 1)`.

        Returns:
            torch.Tensor: reconstructed & predicted results with shape `(B, D, T + i)`.
        """
        return self.mlp_z(rep)
