import math
from typing import Tuple, Union

import torch
from torch import nn


class DAF(nn.Module):
    """DAF

    Args:
        feat_dim (int): :math:`d`.
        pred_len (int): :math:`t`.
        hidden_dim (Union[int, Tuple[int, int]], optional): :math:`h`. Defaults to (64, 64).
        kernel_size (Union[int, Tuple[int, int]], optional): :math:`s`. Defaults to (3, 5).
    """

    def __init__(
        self,
        feat_dim: int,
        pred_len: int,
        hidden_dim: Union[int, Tuple[int, int]] = (64, 64),
        kernel_size: Union[int, Tuple[int, int]] = (3, 5),
    ) -> None:
        super().__init__()
        assert feat_dim % 2 == 0, "Feature dimension must be even"
        self.pred_len = pred_len

        self.attn = SharedAttention(feat_dim, hidden_dim)
        self.src_enc = Encoder(feat_dim, hidden_dim, kernel_size)
        self.src_dec = Decoder(feat_dim, hidden_dim)
        self.tgt_enc = Encoder(feat_dim, hidden_dim, kernel_size)
        self.tgt_dec = Decoder(feat_dim, hidden_dim)

    def forward(
        self, src_data: torch.Tensor, tgt_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Domain Adaptation Forecaster

        Args:
            src_data (torch.Tensor): input tensor of shape `(B, D, T)`, where `D = 1 + d`.
            tgt_data (torch.Tensor): input tensor of shape `(B, D, T)`.

        Returns:
            torch.Tensor: output tensor of shape `(B, D, T)`.
        """
        for _ in range(self.pred_len):
            # for source domain
            src_pattern, src_value = self.src_enc(src_data)
            src_rep = self.attn(src_pattern, src_value)
            src_pred = self.src_dec(src_rep)
            src_data = torch.cat([src_data, src_pred[..., -1]], dim=-1)
            # for target domain
            tgt_pattern, tgt_value = self.tgt_enc(tgt_data)
            tgt_rep = self.attn(tgt_pattern, tgt_value)
            tgt_pred = self.tgt_dec(tgt_rep)
            tgt_data = torch.cat([tgt_data, tgt_pred[..., -1]], dim=-1)
        return src_pred, tgt_pred


class SharedAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared Attention

        Args:
            pattern (torch.Tensor): pattern embedding of shape `(B, M, T)`.

        Returns:
            torch.Tensor: representation of shape `(B, D, T + t)`.
        """
        query, key = self.mlp_q(pattern), self.mlp_k(pattern)
        # interpolation mode
        attn_in = self.calc_attn(query, key, value)
        rep_in = self.mlp_o(attn_in)
        # extrapolation mode
        attn_ex = self.calc_attn(query, key, value, kernel_size=kernel_size)
        rep_ex = self.mlp_o(attn_ex)
        return torch.cat([rep_in, rep_ex], dim=-1)

    def calc_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if isinstance(kernel_size, tuple):
            kernel_size = max(kernel_size)
        if kernel_size:
            coord_unit = (kernel_size - 1) // 2
            query = query[..., -1 - coord_unit : -1 - coord_unit + 1]
            key = key[
                ..., kernel_size - coord_unit - 1 : -1 - coord_unit - 1
            ]  # ? why [..., 96]
            value = value[..., kernel_size:-1]
        attn_logits = torch.exp(
            query.transpose(1, 2) @ key
            - (query.transpose(1, 2) @ key)
            * torch.eye(query.shape[-1])
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
            input_dim=1 + feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )
        self.conv = ConvLayer(
            input_dim=1 + feat_dim,
            output_dim=feat_dim,
            hidden_dim=hidden_dim[0] if isinstance(hidden_dim, tuple) else hidden_dim,
            kernel_size=kernel_size,
        )

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Private Encoders

        Args:
            data (torch.Tensor): input tensor of shape `(B, D, T)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - value of shape `(B, D, T)`.
                - pattern embedding of shape `(B, M, T)`.
        """
        pattern = self.conv(data)
        value = self.mlp_v(data)
        return pattern, value


class Decoder(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """Private Decoders

        Args:
            rep (torch.Tensor): representation of shape `(B, D, T + t)`.

        Returns:
            torch.Tensor: output tensor of shape `(B, D, T + t)`.
        """
        pred = self.mlp(rep)
        return pred


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Union[int, Tuple[int, int]],
        activ: nn.Module = nn.LeakyReLU(),
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, tuple):
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim[0]),
                nn.Linear(hidden_dim[0], hidden_dim[-1]),
                nn.Linear(hidden_dim[-1], output_dim),
                activ,
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Linear(hidden_dim, output_dim),
                activ,
            )

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
        if self.concat:
            return torch.cat([net(x) for net in self.net], dim=1)
        else:
            return self.net(x)


if __name__ == "__main__":
    data = torch.randn(batch_size := 2, feat_dim := 4, hist_len := 100)
    pred_len, hidden_dim, kernel_size = 70, (64, 64), (3, 5)
    print(
        "model hyperparameters:"
        f"\nbatch_size: {batch_size}"
        f"\nhist_len: {hist_len}"
        f"\nfeat_dim: {feat_dim}"
        f"\npred_len: {pred_len}"
        f"\nhidden_dim: {hidden_dim}"
        f"\nkernel_size: {kernel_size}"
    )

    encoder = Encoder(feat_dim, hidden_dim, kernel_size)
    pattern, value = encoder(data)
    print(
        "model output shape:"
        f"\npattern.shape: {pattern.shape}"
        f"\nvalue.shape: {value.shape}"
    )

    attention = SharedAttention(feat_dim, hidden_dim)
    rep = attention(pattern, value)
    print(f"rep.shape: {rep.shape}")

    decoder = Decoder(feat_dim, hidden_dim)
    pred = decoder(rep)
    print(f"pred.shape: {pred.shape}")
