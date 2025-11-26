import os
import time
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, TimeDistributed, Activation, Lambda, Dropout,
    RepeatVector, concatenate, multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from src.config import LOOKBACK_PERIOD, NSE_TICKERS
from src.utils import save_preprocessing, load_preprocessing

def safe_download(ticker):
    """Download a single ticker with retries."""
    for attempt in range(5):
        try:
            df = yf.download(ticker, period="5y", interval="1d", progress=False)
            if not df.empty:
                print(f"Downloaded {ticker} ({len(df)} rows)")
                return df
        except Exception as e:
            print(f"[{ticker}] Retry {attempt+1}/5 due to error: {e}")
        time.sleep(1)

    print(f"[FAILED] Could not download {ticker}")
    return None

def download_and_preprocess_data(tickers, lookback=LOOKBACK_PERIOD):
    print("\nStarting per-ticker download...\n")

    stock_data = {}
    for ticker in tickers:
        df = safe_download(ticker)
        if df is not None and "Close" in df.columns:
            df = df.ffill().bfill()
            stock_data[ticker] = df["Close"]
        else:
            print(f"Skipping {ticker} due to empty or invalid data.")

    if len(stock_data) == 0:
        raise RuntimeError("❌ ALL TICKERS FAILED. No data to train on.")

    df_all = pd.DataFrame(stock_data)
    df_all = df_all.ffill().bfill()

    print(f"\nCombined Dataframe shape: {df_all.shape}\n")

    X_sequences, Y_targets = [], []
    preprocessing_info = {}
    ticker_map = []

    for ticker in df_all.columns:
        series = df_all[ticker].values.reshape(-1, 1)

        if len(series) <= lookback:
            print(f"[SKIP] Not enough data for {ticker}")
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series)

        X, Y = [], []
        for i in range(len(scaled) - lookback):
            X.append(scaled[i:i + lookback])
            Y.append(scaled[i + lookback])

        X = np.array(X)
        Y = np.array(Y)

        preprocessing_info[ticker] = {
            "scaler": scaler,
            "Y": Y
        }

        X_sequences.append(X)
        Y_targets.append(Y)
        ticker_map.extend([ticker] * len(X))

    if len(X_sequences) == 0:
        raise RuntimeError("❌ NO VALID TRAINING SAMPLES AVAILABLE.")

    X_comb = np.concatenate(X_sequences)
    Y_comb = np.concatenate(Y_targets)
    ticker_map = np.array(ticker_map)

    print(f"Total valid samples: {len(X_comb)}\n")

    return X_comb, Y_comb, ticker_map, preprocessing_info

def build_attention_lstm_model(input_shape):
    reg = regularizers.l1_l2(l1=1e-5, l2=1e-4)

    inp = Input(shape=input_shape)

    x = LSTM(128, return_sequences=True, kernel_regularizer=reg)(inp)
    x = Dropout(0.2)(x)

    x = LSTM(64, return_sequences=True, kernel_regularizer=reg)(x)
    x = Dropout(0.2)(x)

    att_score = TimeDistributed(Dense(1, activation="tanh"))(x)
    att_score = Lambda(lambda z: tf.squeeze(z, -1))(att_score)
    att_weights = Activation("softmax")(att_score)
    att_weights = Lambda(lambda z: tf.expand_dims(z, -1))(att_weights)

    context = multiply([x, att_weights])
    context = Lambda(lambda z: tf.reduce_sum(z, axis=1))(context)

    context_rep = RepeatVector(input_shape[0])(context)
    merged = concatenate([x, context_rep], axis=-1)

    x = LSTM(32, return_sequences=False, kernel_regularizer=reg)(merged)
    x = Dropout(0.15)(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    return model

def weighted_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.7 * mse + 0.3 * mae

def train_model(model, X_train, Y_train):
    model.compile(
        optimizer=Adam(0.001),
        loss=weighted_loss,
        metrics=["mae"]
    )

    os.makedirs("models", exist_ok=True)

    cb = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint("models/best_model.h5", save_best_only=True)
    ]

    model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=cb
    )

def evaluate_model(model, X_test, Y_test, ticker_map, prep):
    preds_scaled = model.predict(X_test)

    actual = []
    predicted = []

    for ticker in np.unique(ticker_map):
        idx = np.where(ticker_map == ticker)[0]
        scaler = prep[ticker]["scaler"]

        y_true_inv = scaler.inverse_transform(Y_test[idx])
        y_pred_inv = scaler.inverse_transform(preds_scaled[idx])

        actual.extend(y_true_inv.flatten())
        predicted.extend(y_pred_inv.flatten())

    actual = np.array(actual)
    predicted = np.array(predicted)

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f}")

    return actual, predicted

def main():
    X, Y, tickers, prep = download_and_preprocess_data(NSE_TICKERS)

    X_train, X_test, Y_train, Y_test, t_train, t_test = train_test_split(
        X, Y, tickers, test_size=0.2, shuffle=True, random_state=42
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    model = build_attention_lstm_model((LOOKBACK_PERIOD, 1))
    train_model(model, X_train, Y_train)

    evaluate_model(model, X_test, Y_test, t_test, prep)
    
    save_preprocessing(prep, "models/preprocessing_info.joblib")

if __name__ == "__main__":
    main()
