import pickle
from pathlib import Path
from typing import Tuple, Union

import numpy as np

DATA_ROOT = Path(__file__).parent.resolve()


def make_coldstart(
    tgt_hist_lens: Union[Tuple[int], int],
    data_num: Union[int, float] = 5e3,
    src_hist_len: int = 144,
    pred_len: int = 18,
):
    data_num = int(data_num)
    if isinstance(tgt_hist_lens, int):
        tgt_hist_lens = (tgt_hist_lens,)
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
    with open(DATA_ROOT / "synthetic" / "coldstart_source.pkl", "wb") as f:
        pickle.dump(src_data, f)

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
            DATA_ROOT / "synthetic" / f"coldstart_target_train_{tgt_hist_lens[i]}.pkl",
            "wb",
        ) as f:
            pickle.dump(tgt_data[: -data_num // 5], f)
        with open(
            DATA_ROOT / "synthetic" / f"coldstart_target_valid_{tgt_hist_lens[i]}.pkl",
            "wb",
        ) as f:
            pickle.dump(tgt_data[-data_num // 5 :], f)


def make_fewshot(
    tgt_data_nums: Union[Tuple[int], int],
    src_data_num: Union[int, float] = 5e3,
    hist_len: int = 144,
    pred_len: int = 18,
):
    src_data_num = int(src_data_num)
    if isinstance(tgt_data_nums, int):
        tgt_data_nums = (tgt_data_nums,)
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
    with open(DATA_ROOT / "synthetic" / "fewshot_source.pkl", "wb") as f:
        pickle.dump(src_data, f)

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
            DATA_ROOT / "synthetic" / f"fewshot_target_train_{tgt_data_nums[i]}.pkl",
            "wb",
        ) as f:
            pickle.dump(tgt_data[: -tgt_data_num // 5], f)
        with open(
            DATA_ROOT / "synthetic" / f"fewshot_target_valid_{tgt_data_nums[i]}.pkl",
            "wb",
        ) as f:
            pickle.dump(tgt_data[-tgt_data_num // 5 :], f)


MAKE_DATA = {"coldstart": make_coldstart, "fewshot": make_fewshot}


def main():
    make_coldstart(tgt_hist_lens=[36, 45, 54])
    make_fewshot(tgt_data_nums=[20, 50, 100])


if __name__ == "__main__":
    main()
