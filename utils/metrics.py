from __future__ import annotations
from typing import Callable, Dict

import torch
import matplotlib.pyplot as plt


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_true - y_pred))


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def plot_losses(history: Dict[str, list]):
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["valid_loss"], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()


def plot_predictions(y_true: torch.Tensor, y_pred: torch.Tensor):
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.show()


def permutation_importance(
    model,
    loader,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mae,
    device: str = "cpu",
) -> Dict[int, float]:
    """Simple permutation importance for numeric columns."""
    model.eval()
    base_preds = []
    base_targets = []
    with torch.no_grad():
        for x_num, x_cat, x_seq, y in loader:
            x_num, x_cat, x_seq, y = x_num.to(device), x_cat.to(device), x_seq.to(device), y.to(device)
            base_preds.append(model(x_num, x_cat, x_seq))
            base_targets.append(y)
    base_score = metric(torch.cat(base_targets), torch.cat(base_preds)).item()
    feat_count = next(iter(loader))[0].shape[1]
    importances = {}
    for col in range(feat_count):
        preds = []
        with torch.no_grad():
            for x_num, x_cat, x_seq, y in loader:
                x_num = x_num.clone()
                perm = torch.randperm(x_num.size(0))
                x_num[:, col] = x_num[perm, col]
                x_num, x_cat, x_seq, y = x_num.to(device), x_cat.to(device), x_seq.to(device), y.to(device)
                preds.append(model(x_num, x_cat, x_seq))
        score = metric(torch.cat(base_targets), torch.cat(preds)).item()
        importances[col] = score - base_score
    return importances
