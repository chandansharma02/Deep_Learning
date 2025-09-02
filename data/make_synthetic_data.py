import numpy as np
import pandas as pd
import torch
from pathlib import Path

np.random.seed(42)
torch.manual_seed(42)

N_TOTAL = 10000
SEQ_LEN = 12

specialties = ["cardiology", "endocrinology", "infectious_disease", "oncology", "internal_medicine"]
regions = ["East", "West", "North", "South"]
payer_mix_bucket = ["Medicare-heavy", "Commercial-heavy", "Medicaid-heavy", "Balanced"]
tier_access = ["Open", "Preferred", "Non-preferred", "Restricted"]


def generate_dataframe(n: int) -> pd.DataFrame:
    df = pd.DataFrame({
        "hcp_id": [f"hcp_{i}" for i in range(n)],
        "specialty": np.random.choice(specialties, n),
        "region": np.random.choice(regions, n),
        "payer_mix_bucket": np.random.choice(payer_mix_bucket, n),
        "tier_access": np.random.choice(tier_access, n),
        "rx_baseline": np.random.gamma(5., 20., n),
        "patient_volume": np.random.normal(200., 50., n),
        "naive_patient_share": np.random.rand(n),
        "switch_patient_share": np.random.rand(n),
        "avg_visit_gap_days": np.random.normal(30., 5., n),
        "rep_calls_4w": np.random.poisson(2., n),
        "emails_4w": np.random.poisson(5., n),
        "samples_4w": np.random.poisson(1., n),
        "events_12w": np.random.poisson(0.5, n),
    })

    sales_base = df["rx_baseline"].values * 0.1 + df["patient_volume"].values * 0.05
    noise = np.random.normal(scale=5., size=(n, SEQ_LEN))
    sales_seq = np.maximum(0, sales_base[:, None] + noise)
    calls_seq = np.random.poisson(1.0, size=(n, SEQ_LEN))

    for i in range(SEQ_LEN):
        df[f"sales_w{i+1}"] = sales_seq[:, i]
        df[f"calls_w{i+1}"] = calls_seq[:, i]

    cat_effect = df["tier_access"].map({"Open": 1.2, "Preferred": 1.1, "Non-preferred": 0.9, "Restricted": 0.7}).values
    interaction_effect = 1 + 0.05 * df["rep_calls_4w"].values + 0.02 * df["emails_4w"].values
    recent_sales = sales_seq[:, -4:].sum(axis=1)
    df["future_4w_sales"] = recent_sales * cat_effect * interaction_effect + np.random.normal(0, 10, n)

    return df


def main():
    out_dir = Path(__file__).resolve().parent
    df = generate_dataframe(N_TOTAL)
    train_df = df.iloc[:8000].reset_index(drop=True)
    valid_df = df.iloc[8000:9000].reset_index(drop=True)
    test_df = df.iloc[9000:].reset_index(drop=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    valid_df.to_csv(out_dir / "valid.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print("Generated train.csv, valid.csv, test.csv in", out_dir)


if __name__ == "__main__":
    main()
