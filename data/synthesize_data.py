import pickle
from pathlib import Path
from typing import Tuple, Union

import numpy as np

DATA_ROOT = Path(__file__).parent.resolve()


def make_coldstart(
    data_num: Union[int, float] = 5e3,
    src_hist_len: int = 144,
    tgt_hist_lens: Tuple[int] = [36, 45, 54],
    pred_len: int = 18,
):
    data_num = int(data_num)
    tgt_diversity = len(tgt_hist_lens)
    src_data = []
    for _ in range(data_num):
        src_data.append(
            np.random.uniform(0.5, 5.0)
            * np.sin(
                np.arange(src_hist_len + pred_len)
                * (2 * np.pi / np.random.uniform(144 / 20, 144))
                + np.random.uniform(-2.0, 2.0) * np.pi
            )
            + np.random.uniform(-3.0, 3.0)
            + np.random.normal(0, 0.2)
        )
    with open(DATA_ROOT / "coldstart_source_train.pkl", "wb") as f:
        pickle.dump(src_data[: -data_num // 5], f)
    with open(DATA_ROOT / "coldstart_source_valid.pkl", "wb") as f:
        pickle.dump(src_data[-data_num // 5 :], f)

    for i in range(tgt_diversity):
        tgt_hist_len = tgt_hist_lens[i]
        tgt_data = []
        for _ in range(data_num):
            tgt_data.append(
                np.random.uniform(0.5, 5.0)
                * np.sin(
                    np.arange(tgt_hist_len + pred_len) * (2 * np.pi / 36)
                    + np.random.uniform(-2.0, 2.0) * np.pi
                )
                + np.random.uniform(-3.0, 3.0)
                + np.random.normal(0, 0.2)
            )
        with open(
            DATA_ROOT / f"coldstart_target_train_{tgt_hist_lens[i]}.pkl", "wb"
        ) as f:
            pickle.dump(tgt_data[-data_num // 5 :], f)
        with open(
            DATA_ROOT / f"coldstart_target_valid_{tgt_hist_lens[i]}.pkl", "wb"
        ) as f:
            pickle.dump(tgt_data[: -data_num // 5], f)


def make_fewshot(
    src_data_num: Union[int, float] = 5e3,
    tgt_data_nums: Tuple[int] = [20, 50, 100],
    hist_len: int = 144,
    pred_len: int = 18,
):
    src_data_num = int(src_data_num)
    tgt_diversity = len(tgt_data_nums)
    src_data = []
    for _ in range(src_data_num):
        src_data.append(
            np.random.uniform(0.5, 5.0)
            * np.sin(
                np.arange(hist_len + pred_len)
                * (2 * np.pi / np.random.uniform(144 / 20, 144))
                + np.random.uniform(-2.0, 2.0) * np.pi
            )
            + np.random.uniform(-3.0, 3.0)
            + np.random.normal(0, 0.2)
        )
    with open(DATA_ROOT / "fewshot_source_train.pkl", "wb") as f:
        pickle.dump(src_data[: -src_data_num // 5], f)
    with open(DATA_ROOT / "fewshot_source_valid.pkl", "wb") as f:
        pickle.dump(src_data[-src_data_num // 5 :], f)

    for i in range(tgt_diversity):
        tgt_data_num = tgt_data_nums[i]
        tgt_data = []
        for _ in range(tgt_data_num):
            tgt_data.append(
                np.random.uniform(0.5, 5.0)
                * np.sin(
                    np.arange(hist_len + pred_len)
                    * (2 * np.pi / np.random.uniform(144 / 20, 144))
                    + np.random.uniform(-2.0, 2.0) * np.pi
                )
                + np.random.uniform(-3.0, 3.0)
                + np.random.normal(0, 0.2)
            )
        with open(
            DATA_ROOT / f"fewshot_target_train_{tgt_data_nums[i]}.pkl", "wb"
        ) as f:
            pickle.dump(tgt_data[: -tgt_data_num // 5], f)
        with open(
            DATA_ROOT / f"fewshot_target_valid_{tgt_data_nums[i]}.pkl", "wb"
        ) as f:
            pickle.dump(tgt_data[-tgt_data_num // 5 :], f)


if __name__ == "__main__":
    for make in [make_coldstart, make_fewshot]:
        make()
