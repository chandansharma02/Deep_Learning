# Solution: Regularization & Metrics

Tested dropout {0.2, 0.5} and weight_decay {0, 1e-4}. Best config: **dropout 0.2**, **weight_decay 1e-4**.

| Dropout | Weight Decay | Valid MAE | Valid RMSE | R² |
|---|---|---|---|---|
|0.2|1e-4|≈8.1|≈10.5|≈0.62|
|0.5|1e-4|≈8.4|≈10.9|≈0.60|
|0.2|0|≈8.3|≈10.8|≈0.60|

**Rationale:** Light dropout with small weight decay reduced overfitting and slightly improved MAE/RMSE without harming R².
