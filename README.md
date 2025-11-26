# LSTM Market Dynamics

> Advanced end-to-end example for multi-stock LSTM forecasting (attention-enhanced), training on historical daily prices via `yfinance`.

**Status:** Ready-to-run prototype — honest note: this repo is a strong starting point but not production-ready. Expect to iterate on data handling, backtesting, and deployment before using for live trading.


---

## Quick links
- Implementation: `src/lstm_train.py`. fileciteturn0file0
- Model checkpoint output: `./models/best_model.h5` (created by training).

---

## What this project does (short)
This repository downloads 5 years of daily stock prices (NSE tickers listed in the script), scales and sequences them into lookback windows, trains an LSTM-based sequence model with a simple attention mechanism, and visualizes predictions against ground truth. The model uses a combined weighted MSE+MAE loss and saves the best model to disk.

Key pipeline steps (see code): download → clean → scale → create sequences → train → evaluate → visualise. fileciteturn0file0

---

## Features & highlights
- Multi-stock batching: builds sequences for every ticker and trains one model on the pooled dataset. fileciteturn0file0
- Attention-augmented stacked LSTM architecture (`build_attention_lstm_model`). fileciteturn0file0
- Custom weighted loss combining MSE and MAE (`weighted_mse_mae_loss`) for robustness to outliers. fileciteturn0file0
- Early stopping + best-model checkpointing for stable training. fileciteturn0file0

---

## Repository structure (recommended)
```
lstm-market-dynamics/
├─ src/
│  └─ lstm_train.py        # main implementation (you have this file)
├─ requirements.txt        # (add below dependencies)
├─ models/                 # model checkpoints saved here after training
├─ README.md               # <-- this file
└─ LICENSE                 # MIT license
```

> If your file is named `lstm_trin.py` (typo), rename it to `lstm_train.py` to match the instructions and examples above.

---

## Requirements (suggested)
Create a `requirements.txt` with at minimum the following dependencies (compatible versions are suggestions — pin as needed):

```
python>=3.9
numpy
pandas
matplotlib
scikit-learn
yfinance
tensorflow>=2.10
```
Install with:
```bash
python -m pip install -r requirements.txt
```

---

## Usage (run locally)
1. Clone the repo and place `lstm_train.py` inside `src/` (or run from root adjusting paths).
2. Create a `models/` folder (or let the script create it):
```bash
mkdir -p models
```
3. Run training (from repo root):
```bash
python src/lstm_train.py
```

The script will:
- Download price data using `yfinance` (5 years, daily).
- Preprocess, scale and create lookback sequences (default `LOOKBACK_PERIOD = 60`).
- Train model and save the best checkpoint to `./models/best_model.h5`.
- Print MSE/MAE/MAPE and display example prediction plots. fileciteturn0file0

---

## Configuration (quick reference)
You can edit hyperparameters directly in the script near the top:
- `LOOKBACK_PERIOD` — window length for sequences (default `60`). fileciteturn0file0
- `NSE_TICKERS` — list of tickers to download and train on. fileciteturn0file0
- Training schedule / optimizer / callbacks are defined in `train_model`. fileciteturn0file0

---

## How the model works (brief, read the code for details)
- Model: stacked LSTMs (128 → 64 → 32 units) with dropout and L1/L2 regularization. Attention weights are computed over timesteps and used to create a context vector concatenated back with LSTM outputs before the final LSTM and dense heads. See `build_attention_lstm_model`. fileciteturn0file0
- Loss: `0.7*MSE + 0.3*MAE` implemented in `weighted_mse_mae_loss` to balance large and small errors. fileciteturn0file0
- Optimizer: Adam with exponential decay learning rate schedule and gradient clipping. fileciteturn0file0

---

## Output and evaluation
- Best model saved to `./models/best_model.h5`. fileciteturn0file0
- Evaluation prints MSE, MAE and MAPE (computed after inverse-scaling per ticker), and `visualize_predictions` plots sample tickers' predicted vs actual prices. fileciteturn0file0

---

## Notes, known limitations & suggested fixes (be direct)
1. **Data leakage risk**: pooling sequences from multiple tickers and shuffling can leak temporal structure across tickers if not careful. Consider group-wise time-ordered train/test splits or walk-forward validation for backtesting.
2. **Scaling per-ticker**: good (each ticker uses its own `MinMaxScaler`), but when pooling data for a single model, you may want to include additional features (volume, returns, technical indicators) for richer signals. fileciteturn0file0
3. **Backtesting missing**: there is no trading simulator or transaction cost model. Do not use raw model outputs for live trading without a proper backtest and risk controls.
4. **Performance and memory**: concatenating all sequences for many tickers can create large arrays. If you see memory issues, switch to a generator or TF `tf.data.Dataset` pipeline that yields per-stock or batched sequences on the fly.
5. **Reproducibility**: set random seeds (numpy, tensorflow, python) and record package versions in `requirements.txt` to get repeatable runs.
6. **Productionization**: to deploy, wrap preprocessing and scaler objects for each ticker into a saved preprocessing pipeline (e.g., `joblib` + model).

---

## Recommended improvements (order by impact)
1. Implement time-series cross-validation (rolling-window) for realistic model evaluation.
2. Add technical features (rolling means, momentum, volatility) and categorical embeddings for sector/stock id.
3. Replace `yfinance` batch download with robust data source or local storage to avoid API hiccups/rate limits.
4. Move model training to `tf.data` pipelines to reduce memory footprint and support larger corpora.
5. Add unit tests for preprocessing and a reproducible training script with explicit seeds.

---

## Contributing
- Create issues for bugs or feature requests.
- Submit PRs against `main`. Keep changes small and focused.

---

## License
This project is released under the **MIT License** — see `LICENSE` in the repo for details.

---

### Contact
If you want a quick audit of the repo and a prioritized TODO list, tell me and I’ll give a direct, prioritized list — no sugarcoating.
