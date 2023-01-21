import torch


def calc_nd(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Calculate ND

    Args:
        true (torch.Tensor): ground truth with shape `(B, D, t)`.
        pred (torch.Tensor): prediction with shape `(B, D, t)`.

    Returns:
        torch.Tensor: normalized deviation.
    """
    return (true - pred).abs().sum() / (true.abs().sum() + 1e-7)
