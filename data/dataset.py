import pickle
from typing import Tuple

import torch
from torch.utils import data as dt


class SyntheticDataset(dt.Dataset):
    def __init__(self, path: str, feat_dim: int, pred_len: int) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.pred_len = pred_len

        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self) -> float:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = torch.FloatTensor(self.data[idx]).reshape(self.feat_dim, -1)
        return item[..., : -self.pred_len], item[..., -self.pred_len :]
