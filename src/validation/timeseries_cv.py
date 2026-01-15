from dataclasses import dataclass
import pandas as pd


@dataclass
class Fold:
    train_idx: pd.Index
    val_idx: pd.Index


def walk_forward_folds(
    index: pd.Index, min_train_size: int, horizon: int, step_size: int, n_folds: int
) -> list[Fold]:
    folds = []
    n = len(index)
    train_end = min_train_size

    for _ in range(n_folds):
        val_start = train_end
        val_end = val_start + horizon
        if val_end > n:
            break
        folds.append(
            Fold(train_idx=index[:train_end], val_idx=index[val_start:val_end])
        )
        train_end += step_size

    return folds
