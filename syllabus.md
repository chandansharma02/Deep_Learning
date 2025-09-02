# Mini-Course Syllabus: Deep Learning for Sales Prediction

## Pre-work (30–45 min total)
1. **Installation Check** – run `prework/00_install_check.ipynb`.
   - Objective: verify Python 3.10+ and CPU-only PyTorch work.
2. **PyTorch Basics** – run `prework/01_pytorch_basics.ipynb`.
   - Objective: tensors, gradients, tiny training loop.
3. **Data Shapes & DataLoaders** – run `prework/02_data_shapes_and_dataloaders.ipynb`.
   - Objective: parse HCP dataset and understand batch shapes.

## Live Session (60 min)
1. **Intro Slides (10 min)** – `session/10_intro_sales_dl_slides.md`
   - Why deep learning vs baselines; pitfalls.
2. **CNN Model (25 min)** – `session/11_cnn_timeseries_sales.ipynb`
   - 1D-CNN over weekly sales + static features.
3. **LSTM Model (25 min)** – `session/12_rnn_lstm_sales.ipynb`
   - LSTM over weekly sales + static features.

### Outcomes
- Build and train PyTorch models on synthetic HCP-level data.
- Compare CNN vs LSTM performance.
- Apply regularization and feature ablation to gauge impact.

*Compute note: all code runs on CPU within a local `venv`; no GPU required.*
