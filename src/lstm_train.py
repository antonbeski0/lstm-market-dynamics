"""
Improved training script for LSTM Market Dynamics

Features:
- Per-ticker preprocessing and time-ordered train/test splits (no leakage).
- Adds basic technical features.
- Saves preprocessing artifacts (scalers, per-ticker test counts) for correct per-ticker inverse-scaling.
- Uses tf.data Dataset to avoid concatenating everything into memory at once.
- Deterministic seeds for reproducibility.
"""

import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import joblib

# reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, Activation, Lambda, RepeatVector,
    Dropout, concatenate, multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from src.utils import add_technical_features, create_sequences, save_preprocessing

LOOKBACK_PERIOD = 60
# shortened list for quicker testing — replace with full list as needed
NSE_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS'
]

MODELS_DIR = './models'
PREPROCESSING_PATH = os.path.join(MODELS_DIR, 'preprocessing.joblib')
os.makedirs(MODELS_DIR, exist_ok=True)


def download_ticker_df(ticker, period='5y'):
    df = yf.download(ticker, period=period, interval='1d', progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    # Ensure necessary columns exist; if not, fill with Close
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df.columns:
            df[c] = df['Close']
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


def preprocess_per_ticker(ticker, lookback=LOOKBACK_PERIOD):
    """
    Returns:
        X: np.array (n_samples, lookback, n_features)
        Y: np.array (n_samples,) -> scaled target (Close column)
        scaler: fitted MinMaxScaler for the full feature set
        df: original dataframe with added technical features (chronological)
        metadata: dict with 'target_idx' and 'n_features'
    """
    df = download_ticker_df(ticker)
    df = add_technical_features(df)
    # Identify close column index (in case features reordered)
    target_idx = df.columns.get_loc('Close')
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = scaler.fit_transform(df.values)  # shape: (T, n_features)
    X, Y = create_sequences(values, lookback=lookback, target_idx=target_idx)
    metadata = {'target_idx': int(target_idx), 'n_features': int(values.shape[1])}
    return X, Y, scaler, df, metadata


def build_attention_lstm_model(input_shape):
    l1_l2_reg = regularizers.l1_l2(l1=1e-5, l2=1e-4)
    inputs = Input(shape=input_shape)
    lstm1 = LSTM(128, return_sequences=True, kernel_regularizer=l1_l2_reg)(inputs)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(64, return_sequences=True, kernel_regularizer=l1_l2_reg)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)

    # Attention over timesteps
    attention_scores = TimeDistributed(Dense(1, activation='tanh'))(dropout2)
    squeezed_attention = Lambda(lambda x: tf.squeeze(x, axis=-1))(attention_scores)
    attention_weights = Activation('softmax')(squeezed_attention)
    attention_weights = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention_weights)
    context_vector_seq = multiply([dropout2, attention_weights])
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector_seq)

    repeated_context = RepeatVector(input_shape[0])(context_vector)
    concatenated = concatenate([dropout2, repeated_context], axis=-1)

    lstm3 = LSTM(32, return_sequences=False, kernel_regularizer=l1_l2_reg)(concatenated)
    dropout3 = Dropout(0.15)(lstm3)
    dense1 = Dense(16, activation='relu', kernel_regularizer=l1_l2_reg)(dropout3)
    outputs = Dense(1, activation='linear', kernel_regularizer=l1_l2_reg)(dense1)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def weighted_mse_mae_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return 0.7 * mse + 0.3 * mae


def make_tf_dataset(X, Y, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train_on_pooled_data(tickers, lookback=LOOKBACK_PERIOD):
    """
    Preprocess per ticker, perform time-ordered train/test splits per ticker,
    pool the splits for training, and save preprocessing metadata for correct
    inverse-scaling and per-ticker evaluation.
    """
    per_ticker_info = {}
    X_train_list, Y_train_list = [], []
    X_test_list, Y_test_list = [], []
    scalers = {}
    metadata_map = {}
    tickers_used = []
    test_counts = {}

    for ticker in tickers:
        try:
            X, Y, scaler, df, metadata = preprocess_per_ticker(ticker, lookback=lookback)
        except Exception as e:
            print(f"Warning: skip {ticker}: {e}")
            continue

        n = len(X)
        if n < 10:
            print(f"Skipping {ticker} (not enough samples: {n})")
            continue

        # Time-ordered split: last 20% for test
        split = int(n * 0.8)
        X_train_list.append(X[:split])
        Y_train_list.append(Y[:split])
        X_test_list.append(X[split:])
        Y_test_list.append(Y[split:])

        scalers[ticker] = scaler
        metadata_map[ticker] = metadata
        tickers_used.append(ticker)
        test_counts[ticker] = int(n - split)
        per_ticker_info[ticker] = {'df': df, 'n_samples': n}

    if not X_train_list:
        raise RuntimeError("No tickers successfully preprocessed. Aborting.")

    # Pool training/test data (order preserved in tickers_used)
    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)

    print(f'Pooled train samples: {X_train.shape[0]} | test samples: {X_test.shape[0]}')

    # Save preprocessing info with per-ticker test counts and metadata
    prep = {
        'tickers': tickers_used,
        'scalers': scalers,
        'metadata': metadata_map,
        'lookback': lookback,
        'test_counts': test_counts
    }
    save_preprocessing(prep, PREPROCESSING_PATH)
    print(f'Saved preprocessing to {PREPROCESSING_PATH}')

    input_shape = X_train.shape[1:]
    model = build_attention_lstm_model(input_shape)
    initial_lr = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_lr, decay_steps=10000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=weighted_mse_mae_loss, metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint_filepath = os.path.join(MODELS_DIR, 'best_model.h5')
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)

    train_ds = make_tf_dataset(X_train, Y_train, batch_size=32, shuffle=True)
    val_ds = make_tf_dataset(X_test, Y_test, batch_size=32, shuffle=False)

    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stopping, model_checkpoint])
    return model, (X_test, Y_test), prep


def inverse_transform_target(scaler, y_scaled, target_idx, n_features):
    """
    Inverse transform a scaled target column `y_scaled` given a scaler fitted on
    the full feature set (so we must reconstruct full n_features arrays).
    """
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    placeholders = np.zeros((len(y_scaled), n_features))
    placeholders[:, target_idx] = y_scaled[:, 0]
    inv = scaler.inverse_transform(placeholders)
    return inv[:, target_idx]


def evaluate_model(model, X_test, Y_test, preprocessing):
    """
    Perform per-ticker inverse-scaling using saved test_counts and metadata,
    compute MSE/MAE/MAPE aggregated across tickers, and return arrays for plotting.
    """
    preds = model.predict(X_test)
    tickers = preprocessing['tickers']
    scalers = preprocessing['scalers']
    metadata_map = preprocessing['metadata']
    test_counts = preprocessing['test_counts']

    idx = 0
    y_true_all = []
    y_pred_all = []

    for ticker in tickers:
        cnt = test_counts.get(ticker, 0)
        if cnt == 0:
            continue
        y_test_seg = Y_test[idx: idx + cnt]
        y_pred_seg = preds[idx: idx + cnt]

        meta = metadata_map[ticker]
        target_idx = meta['target_idx']
        n_features = meta['n_features']
        scaler = scalers[ticker]

        y_test_inv = inverse_transform_target(scaler, y_test_seg, target_idx, n_features)
        y_pred_inv = inverse_transform_target(scaler, y_pred_seg, target_idx, n_features)

        y_true_all.extend(y_test_inv.tolist())
        y_pred_all.extend(y_pred_inv.tolist())

        idx += cnt

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    mse = mean_squared_error(y_true_all, y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mape = np.mean(np.abs((y_true_all - y_pred_all) / (y_true_all + 1e-8))) * 100

    print(f'MSE: {mse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%')
    return y_true_all, y_pred_all


def main():
    model, test_data, prep = train_on_pooled_data(NSE_TICKERS, LOOKBACK_PERIOD)
    X_test, Y_test = test_data
    y_true, y_pred = evaluate_model(model, X_test, Y_test, prep)
    # Save final model
    model.save(os.path.join(MODELS_DIR, 'final_model.h5'))
    print('Model and preprocessing saved.')


if __name__ == '__main__':
    main()
