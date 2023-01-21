from typing import Tuple, Union

import torch
from torch import nn

from .layers import MLPLayer


class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.cls_q = MLPLayer(
            input_dim=feat_dim, output_dim=2, hidden_dim=hidden_dim, is_cls=True
        )
        self.cls_k = MLPLayer(
            input_dim=feat_dim, output_dim=2, hidden_dim=hidden_dim, is_cls=True
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Domain Discriminator

        Args:
            query (torch.Tensor): query with shape `(B, D, T + t)`.
            key (torch.Tensor): key with shape `(B, D, T + t)`.

        Returns:
            torch.Tensor: domain classification result with shape`(B, 2, T + t)`.
        """
        dom_query = self.cls_q(query)
        dom_key = self.cls_k(key)
        return dom_query, dom_key
