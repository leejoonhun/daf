import pickle
from typing import Tuple

import torch
import torch.utils.data as dt


class SyntheticDataset(dt.Dataset):
    def __init__(self, path: str, pred_len: int) -> None:
        super().__init__()
        self.pred_len = pred_len

        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self) -> float:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.data[idx][: -self.pred_len]),
            torch.FloatTensor(self.data[idx][-self.pred_len :]),
        )


def get_dataloaders(type: str, type_param: int, pred_len: int, batch_size: int):
    tgt_train_path, tgt_valid_path = (
        f"data/{type}_target_train_{type_param}.pkl",
        f"data/{type}_target_valid_{type_param}.pkl",
    )
    tgt_trainset, tgt_validset = (
        SyntheticDataset(tgt_train_path, pred_len),
        SyntheticDataset(tgt_valid_path, pred_len),
    )
    src_path = f"data/{type}_source.pkl"
    src_dataset = SyntheticDataset(src_path, pred_len)

    src_batch_size = batch_size * (
        len(src_dataset) // (len(tgt_trainset) + len(src_dataset))
    )

    return (
        dt.DataLoader(src_dataset, batch_size=src_batch_size, shuffle=True),
        dt.DataLoader(
            tgt_trainset, batch_size=batch_size - src_batch_size, shuffle=True
        ),
        dt.DataLoader(tgt_validset, batch_size=batch_size, shuffle=False),
    )
