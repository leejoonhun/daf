import pickle
from os import path as osp
from typing import Tuple

import torch
from torch.utils import data as dt

from .synthesize_data import MAKE_DATA


class SyntheticDataset(dt.Dataset):
    def __init__(self, path: str, feat_dim: int, pred_len: int) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.pred_len = pred_len

        if not osp.exists(path):
            syn_type, _, _, syn_param = path.stem.split("_")
            print(f"Making {syn_type}-{syn_param} data before training..")
            MAKE_DATA[syn_type](int(syn_param))
            print("Done\n")
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self) -> float:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = torch.FloatTensor(self.data[idx]).reshape(self.feat_dim, -1)
        return item[..., : -self.pred_len], item[..., -self.pred_len :]
