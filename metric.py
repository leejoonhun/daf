import torch


def calc_nd(true: torch.Tensor, pred: torch.Tensor):
    """Calculate ND

    Args:
        true (torch.Tensor): ground truth with shape `(B, D, t)`.
        pred (torch.Tensor): prediction with shape `(B, D, t)`.
    """
    numerator = (true - pred).abs().sum()
    denominator = true.abs().sum()
    return numerator / denominator
