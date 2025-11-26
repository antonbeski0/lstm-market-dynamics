import os
import random
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
    Input, LSTM, Dense, TimeDistributed, Activation, Lambda, RepeatVector,
    GlobalAveragePooling1D, Dropout, concatenate, multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

# Global parameters
LOOKBACK_PERIOD = 60
NSE_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'HINDUNILVR.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS',
    'BAJFINANCE.NS', 'ASIANPAINT.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
    'SUNPHARMA.NS', 'MARUTI.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'WIPRO.NS', 'TECHM.NS', 'HCLTECH.NS', 'INDUSINDBK.NS', 'TITAN.NS',
    'ADANIPORTS.NS', 'GRASIM.NS', 'POWERGRID.NS', 'NTPC.NS', 'ONGC.NS',
    'COALINDIA.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS', 'MM.NS', 'HEROMOTOCO.NS',
    'DRREDDY.NS', 'CIPLA.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'BRITANNIA.NS',
    'EICHERMOT.NS', 'JSWSTEEL.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 
    'HDFCLTD.NS', 'BAJAJFINSV.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS',
    'BANDHANBNK.NS', 'PIDILITIND.NS', 'DMART.NS'
]

def download_and_preprocess_data(tickers, lookback=LOOKBACK_PERIOD):
    """
    Batch download stock data for given tickers, scale and create sequences.
    Returns combined X, Y, and stock ticker map and preprocessing info dict.
    """
    print("Downloading stock data in batch...")
    data = yf.download(
        tickers=tickers,
        period='5y',
        interval='1d',
        group_by='ticker',
        progress=False
    )
    print("Download complete.")
    
    stock_data = pd.DataFrame()
    for ticker in tickers:
        try:
            prices = data[ticker]['Close'].dropna()
            stock_data[ticker] = prices
        except Exception as e:
            print(f"Failed to get data for {ticker}: {e}")

    # Drop columns with all NaNs and forward fill + linear interpolate others
    stock_data.dropna(axis=1, how='all', inplace=True)
    stock_data.fillna(method='ffill', inplace=True)
    stock_data.interpolate(method='linear', inplace=True)

    print(f"Cleaned stock data with shape: {stock_data.shape}")

    X_sequences, Y_targets = [], []
    stock_preprocessing_info = {}

    for ticker in stock_data.columns:
        prices = stock_data[ticker].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        X, Y = [], []
        for i in range(len(scaled_prices) - lookback):
            X.append(scaled_prices[i:i+lookback])
            Y.append(scaled_prices[i+lookback])
        X, Y = np.array(X), np.array(Y)
        stock_preprocessing_info[ticker] = {
            'original_prices': prices,
            'scaler': scaler,
            'X': X,
            'Y': Y
        }
        X_sequences.append(X)
        Y_targets.append(Y)

    X_combined = np.concatenate(X_sequences, axis=0) if X_sequences else np.array([])
    Y_combined = np.concatenate(Y_targets, axis=0) if Y_targets else np.array([])

    # Create stock ticker map - which stock each sample originates from
    stock_ticker_map_combined = []
    for ticker in stock_preprocessing_info.keys():
        n_samples = len(stock_preprocessing_info[ticker]['X'])
        stock_ticker_map_combined.extend([ticker] * n_samples)
    stock_ticker_map_combined = np.array(stock_ticker_map_combined)

    print(f"Total samples: {X_combined.shape[0]}")

    return X_combined, Y_combined, stock_ticker_map_combined, stock_preprocessing_info

def build_attention_lstm_model(input_shape):
    """
    Build a multi-layer LSTM with attention, dropout and regularization.
    """
    l1_l2_reg = regularizers.l1_l2(l1=1e-5, l2=1e-4)

    inputs = Input(shape=input_shape)

    lstm1 = LSTM(128, return_sequences=True, kernel_regularizer=l1_l2_reg,
                 recurrent_regularizer=l1_l2_reg)(inputs)
    dropout1 = Dropout(0.2)(lstm1)

    lstm2 = LSTM(64, return_sequences=True, kernel_regularizer=l1_l2_reg,
                 recurrent_regularizer=l1_l2_reg)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)

    # Attention mechanism
    attention_scores = TimeDistributed(Dense(1, activation='tanh'))(dropout2)
    # Squeeze last dim for softmax over timesteps
    squeezed_attention = Lambda(lambda x: tf.squeeze(x, axis=-1))(attention_scores)
    attention_weights = Activation('softmax')(squeezed_attention)
    # Expand dims to multiply
    attention_weights = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention_weights)
    context_vector_seq = multiply([dropout2, attention_weights])
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector_seq)

    repeated_context = RepeatVector(input_shape[0])(context_vector)
    concatenated = concatenate([dropout2, repeated_context], axis=-1)

    lstm3 = LSTM(32, return_sequences=False, kernel_regularizer=l1_l2_reg,
                 recurrent_regularizer=l1_l2_reg)(concatenated)
    dropout3 = Dropout(0.15)(lstm3)
    dense1 = Dense(16, activation='relu', kernel_regularizer=l1_l2_reg)(dropout3)
    outputs = Dense(1, activation='linear', kernel_regularizer=l1_l2_reg)(dense1)

    model = Model(inputs=inputs, outputs=outputs)
    print("Model with Attention built successfully.")
    return model

def weighted_mse_mae_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return 0.7 * mse + 0.3 * mae

def train_model(model, X_train, Y_train):
    initial_lr = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=10000, decay_rate=0.9, staircase=True)
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

    model.compile(optimizer=optimizer,
                  loss=weighted_mse_mae_loss,
                  metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint_filepath = './models/best_model.h5'
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, Y_train, validation_split=0.2,
                        epochs=200, batch_size=32,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    return history

def evaluate_model(model, X_test, Y_test, stock_test_map, preprocessing_info):
    """
    Perform predictions, inverse scale per stock, calculate metrics.
    """
    predictions_scaled = model.predict(X_test, verbose=0)
    Y_actual, Y_predicted = [], []

    unique_tickers = np.unique(stock_test_map)
    for ticker in unique_tickers:
        indices = np.where(stock_test_map == ticker)[0]
        scaler = preprocessing_info[ticker]['scaler']

        y_test_scaled = Y_test[indices].reshape(-1, 1)
        y_pred_scaled = predictions_scaled[indices].reshape(-1, 1)

        y_test_inv = scaler.inverse_transform(y_test_scaled).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred_scaled).flatten()

        Y_actual.extend(y_test_inv)
        Y_predicted.extend(y_pred_inv)

    Y_actual = np.array(Y_actual)
    Y_predicted = np.array(Y_predicted)

    mse = mean_squared_error(Y_actual, Y_predicted)
    mae = mean_absolute_error(Y_actual, Y_predicted)
    mape = np.mean(np.abs((Y_actual - Y_predicted) / (Y_actual + 1e-8))) * 100

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    return Y_actual, Y_predicted, unique_tickers

def visualize_predictions(Y_actual, Y_predicted, stock_test_map, unique_tickers, preprocessing_info):
    num_stocks_to_display = min(5, len(unique_tickers))
    selected_tickers = random.sample(list(unique_tickers), num_stocks_to_display)
    print(f'Selected tickers for visualization: {", ".join(selected_tickers)}')

    plt.figure(figsize=(15, 5 * num_stocks_to_display))

    for i, ticker in enumerate(selected_tickers, 1):
        indices = np.where(stock_test_map == ticker)[0]

        y_test_scaled = preprocessing_info[ticker]['Y'][indices]
        y_pred_scaled = Y_predicted[indices]

        scaler = preprocessing_info[ticker]['scaler']

        # Inverse transform for plotting, making sure arrays match shape
        y_test_plot = scaler.inverse_transform(
            preprocessing_info[ticker]['Y'][indices].reshape(-1,1)).flatten()
        y_pred_plot = Y_predicted[indices]

        plt.subplot(num_stocks_to_display, 1, i)
        plt.plot(y_test_plot, label='Actual Prices', color='blue')
        plt.plot(y_pred_plot, label='Predicted Prices', color='red', linestyle='--')
        plt.title(f'Predictions vs Actuals for {ticker}')
        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price')
        plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    X, Y, stock_map, preprocessing_info = download_and_preprocess_data(NSE_TICKERS, LOOKBACK_PERIOD)
    # Train/test split keeping alignment for stock_map info
    X_train, X_test, Y_train, Y_test, stock_train_map, stock_test_map = train_test_split(
        X, Y, stock_map, test_size=0.2, random_state=42, shuffle=True
    )

    print(f'Training samples: {X_train.shape[0]}')
    print(f'Testing samples: {X_test.shape[0]}')

    model = build_attention_lstm_model(input_shape=(LOOKBACK_PERIOD, 1))

    train_model(model, X_train, Y_train)

    Y_actual, Y_predicted, unique_test_tickers = evaluate_model(
        model, X_test, Y_test, stock_test_map, preprocessing_info)

    visualize_predictions(Y_actual, Y_predicted, stock_test_map, unique_test_tickers, preprocessing_info)

if __name__ == '__main__':
    main()
