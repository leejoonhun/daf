from torch.utils import data as dt

from .dataset import SyntheticDataset


def get_dataloaders(
    syn_type: str, syn_param: int, feat_dim: int, pred_len: int, batch_size: int
):
    tgt_train_path, tgt_valid_path = (
        f"data/{syn_type}_target_train_{syn_param}.pkl",
        f"data/{syn_type}_target_valid_{syn_param}.pkl",
    )
    tgt_trainset, tgt_validset = (
        SyntheticDataset(tgt_train_path, feat_dim, pred_len),
        SyntheticDataset(tgt_valid_path, feat_dim, pred_len),
    )
    src_path = f"data/{syn_type}_source.pkl"
    src_dataset = SyntheticDataset(src_path, feat_dim, pred_len)

    return (
        dt.DataLoader(src_dataset, batch_size=batch_size, shuffle=True),
        dt.DataLoader(tgt_trainset, batch_size=batch_size, shuffle=True),
        dt.DataLoader(tgt_validset, batch_size=batch_size, shuffle=False),
    )
