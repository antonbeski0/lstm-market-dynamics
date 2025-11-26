# LSTM-Based Stock Market Prediction with Attention

This project provides a complete framework for predicting stock market movements using a Long Short-Term Memory (LSTM) neural network enhanced with an attention mechanism. It automatically fetches historical stock data, preprocesses it, trains a predictive model, and saves the best-performing version for later use.

## Key Features

- **Automated Data Caching**: Downloads and caches 5 years of historical stock data for a predefined list of tickers from Yahoo Finance.
- **Advanced Preprocessing**: Normalizes and sequences the data into a format suitable for LSTM networks.
- **Attention-Based LSTM Model**: Implements a sophisticated LSTM architecture with an attention layer to focus on the most relevant historical data points.
- **Robust Training**: Includes early stopping and model checkpointing to prevent overfitting and save the best model.
- **Strategy Backtesting**: A simple backtesting script is included to evaluate the model's performance from a trading strategy perspective.
- **Extensible**: The code is modular and can be easily extended to include more tickers, different models, or more complex backtesting strategies.

## Project Structure

```
lstm-market-dynamics/
│
├── .github/
│   └── workflows/
│       └── ci.yml      # Continuous integration workflow
├── models/             # (Auto-created) Stores trained models and scalers
├── src/
│   ├── lstm_train.py   # Main script for data processing and model training
│   ├── backtest.py     # Script for backtesting trading strategies
│   └── utils.py        # Utility functions (currently empty)
├── .gitignore          # Specifies files to be ignored by Git
├── Dockerfile          # Defines a Docker container for the project
├── LICENSE             # Project license
├── README.md           # This file
└── requirements.txt    # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Pip and a virtual environment (`venv`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/antonbeski0/lstm-market-dynamics.git
   cd lstm-market-dynamics
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Train the Model:**

   To start the training process, run the `lstm_train.py` script:

   ```bash
   python src/lstm_train.py
   ```

   The script will:
   - Download the necessary stock data.
   - Preprocess the data and create training/test splits.
   - Build and compile the LSTM model.
   - Train the model, saving the best version to `models/best_model.h5`.
   - Evaluate the final model and print the performance metrics.

2. **Backtesting:**

   The `backtest.py` script contains a function for a simple directional backtest. You can integrate this with the trained model to simulate a trading strategy.

## Model Architecture

The model uses a deep LSTM network with an attention mechanism.

- **Input Layer**: Takes a sequence of historical price data (e.g., the last 60 days).
- **LSTM Layers**: Two LSTM layers with 128 and 64 units, respectively, capture temporal patterns. Dropout is used for regularization.
- **Attention Layer**: A custom attention mechanism is applied to allow the model to weigh the importance of different time steps in the input sequence.
- **Output Layer**: A dense layer that outputs the predicted stock price for the next time step.

The model is trained using a weighted loss function that combines Mean Squared Error (MSE) and Mean Absolute Error (MAE) to provide a balanced optimization target.

## Disclaimer

This project is for educational and research purposes only. The predictions are based on historical data and do not guarantee future performance. Trading in financial markets involves significant risk, and you should not make investment decisions based solely on the output of this model.
