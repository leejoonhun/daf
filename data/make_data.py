import pickle
from pathlib import Path
from typing import Tuple, Union

import numpy as np

DATA_ROOT = Path(__file__).parent.resolve()


def make_coldstart(
    data_num: Union[int, float] = 5e3,
    src_hist_len: int = 144,
    tgt_hist_lens: Tuple[int] = [36, 45, 54],
    tgt_period: int = 36,
):
    tgt_diversity = len(tgt_hist_lens)
    src_data, tgt_data = [], [[] for _ in range(tgt_diversity)]
    for _ in range(int(data_num)):
        src_data.append(
            (
                np.sin(
                    np.arange(src_hist_len)
                    * (2 * np.pi / np.random.randint(1, src_hist_len))
                    + np.random.uniform(-np.pi, np.pi)
                )
                + np.random.uniform(-1, 1)
            )
            * np.random.uniform(1, 1e3)
        )
    with open(DATA_ROOT / "coldstart_source.pkl", "wb") as f:
        pickle.dump(src_data, f)

    for i in range(tgt_diversity):
        tgt_hist_len = tgt_hist_lens[i]
        for _ in range(int(data_num)):
            tgt_data[i].append(
                (
                    np.sin(
                        np.arange(tgt_hist_len)
                        * (2 * np.pi / np.random.randint(1, tgt_hist_len))
                        + np.random.uniform(-np.pi, np.pi)
                    )
                    + np.random.uniform(-1, 1)
                )
                * np.random.uniform(1, 1e3)
            )
        with open(DATA_ROOT / f"coldstart_target_{tgt_hist_lens[i]}.pkl", "wb") as f:
            pickle.dump(tgt_data[i], f)


def make_fewshot(
    src_data_num: Union[int, float] = 5e3,
    tgt_data_nums: Tuple[int] = [20, 50, 100],
    hist_len: int = 144,
):
    tgt_diversity = len(tgt_data_nums)
    src_data, tgt_data = [], [[] for _ in range(tgt_diversity)]
    for _ in range(int(src_data_num)):
        src_data.append(
            (
                np.sin(
                    np.arange(hist_len) * (2 * np.pi / np.random.randint(1, hist_len))
                    + np.random.uniform(-np.pi, np.pi)
                )
                + np.random.uniform(-1, 1)
            )
            * np.random.uniform(1, 1e3)
        )
    with open(DATA_ROOT / "fewshot_source.pkl", "wb") as f:
        pickle.dump(src_data, f)

    for i in range(tgt_diversity):
        tgt_data_num = tgt_data_nums[i]
        for _ in range(int(tgt_data_num)):
            tgt_data[i].append(
                (
                    np.sin(
                        np.arange(hist_len)
                        * (2 * np.pi / np.random.randint(1, hist_len))
                        + np.random.uniform(-np.pi, np.pi)
                    )
                    + np.random.uniform(-1, 1)
                )
                * np.random.uniform(1, 1e3)
            )
        with open(DATA_ROOT / f"fewshot_target_{tgt_data_nums[i]}.pkl", "wb") as f:
            pickle.dump(tgt_data[i], f)


if __name__ == "__main__":
    for make in [make_coldstart, make_fewshot]:
        make()
