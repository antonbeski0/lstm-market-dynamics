# LSTM Market Dynamics

This repository contains an improved and safer structure for training an LSTM-based model on multiple stock tickers.

Key changes from the prototype:
- **Per-ticker preprocessing** and time-ordered train/test split (no leakage).
- Added **basic technical features** (returns, MA, volatility).
- **Preprocessing artifacts saved** for production inference (`models/preprocessing.joblib`).
- Uses `tf.data` pipeline to reduce memory pressure.
- Includes a minimal **backtest** utility for direction-based signal testing.

## Files included
- `src/lstm_train.py` — training pipeline (per-ticker preprocessing, model training).
- `src/utils.py` — helpers: technical indicators, sequence creation, save/load preprocessing.
- `src/backtest.py` — a simple directional backtest utility.
- `requirements.txt` — packages to install.
- `LICENSE` — MIT (included).
- `.gitignore`, `Dockerfile`, and a basic GitHub Actions workflow included.

## Quick start
1. Create a virtual environment and install deps:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
