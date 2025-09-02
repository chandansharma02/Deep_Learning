# Solution: Feature Ablation

Validation MAE when removing interaction features (CNN model):

| Removed Feature | Valid MAE |
|---|---|
|None|8.1|
|rep_calls_4w|8.9|
|emails_4w|8.4|
|samples_4w|8.2|
|events_12w|8.3|

**Implication:** Dropping `rep_calls_4w` hurts most, suggesting recent rep engagement strongly drives short-term sales.
