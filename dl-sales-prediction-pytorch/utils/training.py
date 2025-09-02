from __future__ import annotations
from typing import Dict, List, Tuple
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import mae


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 3,
    device: str = "cpu",
    checkpoint_dir: str = "artifacts",
) -> Dict[str, List[float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    history = {"train_loss": [], "valid_loss": [], "valid_mae": []}
    best_mae = float("inf")
    wait = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_num, x_cat, x_seq, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x_num, x_cat, x_seq, y = x_num.to(device), x_cat.to(device), x_seq.to(device), y.to(device)
            opt.zero_grad()
            preds = model(x_num, x_cat, x_seq)
            loss = crit(preds, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        history["train_loss"].append(float(torch.tensor(train_losses).mean()))

        model.eval()
        val_losses = []
        val_maes = []
        with torch.no_grad():
            for x_num, x_cat, x_seq, y in valid_loader:
                x_num, x_cat, x_seq, y = x_num.to(device), x_cat.to(device), x_seq.to(device), y.to(device)
                preds = model(x_num, x_cat, x_seq)
                loss = crit(preds, y)
                val_losses.append(loss.item())
                val_maes.append(mae(y, preds))
        vloss = float(torch.tensor(val_losses).mean())
        vmae = float(torch.tensor(val_maes).mean())
        history["valid_loss"].append(vloss)
        history["valid_mae"].append(vmae)

        if vmae < best_mae:
            best_mae = vmae
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    return history, best_path
