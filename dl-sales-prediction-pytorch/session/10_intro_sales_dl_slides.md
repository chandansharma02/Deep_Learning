# Intro: Deep Learning for HCP Sales

- **Objective**: predict next 4 weeks of Rx sales per HCP.
- **Why DL?** captures nonlinear interactions & sequential patterns.
- **Data**: static features + 12 weeks of sales & calls.
- **Cautions**: no leakage (use only past data), scale inputs, watch class imbalance.
- **Pipeline**:
  1. Encode categoricals → embeddings
  2. Normalize numerics
  3. Model sequences (CNN or LSTM)
  4. Concatenate + predict future sales
- **Metrics**: MAE, RMSE, R²
- **Next**: build CNN and LSTM models live.
