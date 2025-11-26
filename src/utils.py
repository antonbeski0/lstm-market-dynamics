import numpy as np
import pandas as pd
import joblib

def add_technical_features(df):
    """
    Adds basic technical indicators for the stock DataFrame.
    df must contain: 'Open','High','Low','Close','Volume'
    """
    df = df.copy()

    # Returns
    df['return'] = df['Close'].pct_change().fillna(0)
    df['log_return'] = np.log1p(df['return'])

    # Moving averages
    df['ma_7'] = df['Close'].rolling(7).mean().fillna(method='bfill')
    df['ma_21'] = df['Close'].rolling(21).mean().fillna(method='bfill')

    # Volatility
    df['volatility_21'] = df['return'].rolling(21).std().fillna(method='bfill')

    df.fillna(method='ffill', inplace=True)
    return df


def create_sequences(values, lookback=60, target_idx=0):
    """
    Converts a time-series matrix into sequences.
    values: np.array shape (T, n_features)
    lookback: timesteps per sample
    target_idx: column index to predict
    """
    X, Y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i:i+lookback])
        Y.append(values[i+lookback, target_idx])  # target is close price
    return np.array(X), np.array(Y)


def save_preprocessing(prep, path):
    """Save preprocessing dict using joblib."""
    joblib.dump(prep, path)


def load_preprocessing(path):
    """Load preprocessing dict."""
    return joblib.load(path)
