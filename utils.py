from typing import Tuple

import torch


def make_true_dom(
    src_dom_: torch.Tensor, tgt_dom_: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make true labels for domain classification"""
    src_dom, tgt_dom = (
        torch.zeros_like(src_dom_, device=src_dom_.device),
        torch.zeros_like(tgt_dom_, device=tgt_dom_.device),
    )
    src_dom[:, 0, :], tgt_dom[:, 1, :] = 1, 1
    return src_dom, tgt_dom
