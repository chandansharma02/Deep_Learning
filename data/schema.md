# Synthetic HCP Sales Dataset Schema

| Column | Type | Notes |
|---|---|---|
|`hcp_id`|string|Unique identifier|
|`specialty`|categorical|{cardiology,endocrinology,infectious_disease,oncology,internal_medicine}|
|`region`|categorical|{East,West,North,South}|
|`payer_mix_bucket`|categorical|{Medicare-heavy,Commercial-heavy,Medicaid-heavy,Balanced}|
|`tier_access`|categorical|{Open,Preferred,Non-preferred,Restricted}|
|`rx_baseline`|float|Avg historic prescriptions|
|`patient_volume`|float|Estimated patient volume|
|`naive_patient_share`|float|0–1 fraction|
|`switch_patient_share`|float|0–1 fraction|
|`avg_visit_gap_days`|float|Average days between visits|
|`rep_calls_4w`|float|Field rep calls in last 4 weeks|
|`emails_4w`|float|Emails sent in last 4 weeks|
|`samples_4w`|float|Samples dropped in last 4 weeks|
|`events_12w`|float|Events attended in last 12 weeks|
|`sales_w1..sales_w12`|float|Weekly sales sequence (most recent first)|
|`calls_w1..calls_w12`|float|Weekly calls sequence|
|`future_4w_sales`|float|Target: weeks 13–16 sales sum|
