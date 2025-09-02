# Deep Learning for Pharma Sales Prediction with PyTorch

Crisp mini-course for practitioners who know Python and basic ML but are new to deep learning with PyTorch.

## Quickstart (Windows, CPU only)
```bash
py -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python .\data\make_synthetic_data.py
jupyter notebook
```
Run the three notebooks in `prework/` first, then open the session notebooks in `session/`.

## Modules and Outcomes
| Module | Notebook/Resource | Time | Outcome |
|---|---|---|---|
|Pre-work 1|`prework/00_install_check.ipynb`|5 min|Verify Python & PyTorch install|
|Pre-work 2|`prework/01_pytorch_basics.ipynb`|15 min|Tensor ops & tiny model|
|Pre-work 3|`prework/02_data_shapes_and_dataloaders.ipynb`|15 min|Understand batch shapes|
|Live Intro|`session/10_intro_sales_dl_slides.md`|10 min|DL vs baselines|
|Live Model 1|`session/11_cnn_timeseries_sales.ipynb`|25 min|1D-CNN for sales|
|Live Model 2|`session/12_rnn_lstm_sales.ipynb`|25 min|LSTM for sales|
|Exercise|`exercises/01_regularization_and_metrics.md`|15 min|Improve generalization|
|Exercise|`exercises/02_feature_ablation.md`|15 min|Feature importance|

## Checklist
- [ ] ✅ Windows venv created and requirements installed
- [ ] ✅ Synthetic data generated (`data/*.csv`)
- [ ] ✅ Pre-work notebooks completed error-free
- [ ] ✅ Ran CNN notebook and captured metrics/plots
- [ ] ✅ Ran LSTM notebook and compared with CNN
- [ ] ✅ Completed at least one exercise and reviewed solution
