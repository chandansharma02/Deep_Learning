from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

CATEGORICAL_COLS = ["specialty", "region", "payer_mix_bucket", "tier_access"]
NUMERIC_COLS = [
    "rx_baseline",
    "patient_volume",
    "naive_patient_share",
    "switch_patient_share",
    "avg_visit_gap_days",
    "rep_calls_4w",
    "emails_4w",
    "samples_4w",
    "events_12w",
]
SALES_COLS = [f"sales_w{i}" for i in range(1, 13)]
CALLS_COLS = [f"calls_w{i}" for i in range(1, 13)]


@dataclass
class Stats:
    cat_maps: Dict[str, Dict[str, int]]
    num_means: pd.Series
    num_stds: pd.Series


class HCPDataset(Dataset):
    """Dataset returning (x_num, x_cat, x_seq, y)."""

    def __init__(self, csv_path: str, stats: Stats | None = None):
        df = pd.read_csv(csv_path)
        self.y = torch.tensor(df["future_4w_sales"].values, dtype=torch.float32)

        # categorical
        if stats is None:
            cat_maps = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))} for c in CATEGORICAL_COLS}
            num_means = df[NUMERIC_COLS].mean()
            num_stds = df[NUMERIC_COLS].std().replace(0, 1)
            self.stats = Stats(cat_maps, num_means, num_stds)
        else:
            self.stats = stats
        self.x_cat = torch.tensor(
            np.stack([df[c].map(self.stats.cat_maps[c]).values for c in CATEGORICAL_COLS], axis=1),
            dtype=torch.long,
        )

        # numeric
        normed = (df[NUMERIC_COLS] - self.stats.num_means) / self.stats.num_stds
        self.x_num = torch.tensor(normed.values, dtype=torch.float32)

        # sequences (N,12,2)
        sales = df[SALES_COLS].values
        calls = df[CALLS_COLS].values
        seq = np.stack([sales, calls], axis=2)
        self.x_seq = torch.tensor(seq, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_cat[idx], self.x_seq[idx], self.y[idx]


def get_dataloaders(data_dir: str, batch_size: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader, Stats]:
    train_ds = HCPDataset(os.path.join(data_dir, "train.csv"))
    stats = train_ds.stats
    valid_ds = HCPDataset(os.path.join(data_dir, "valid.csv"), stats)
    test_ds = HCPDataset(os.path.join(data_dir, "test.csv"), stats)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, valid_loader, test_loader, stats
