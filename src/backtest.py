import numpy as np
import pandas as pd

def simple_directional_backtest(prices, preds, threshold=0.0, pct_cost=0.0005):
    '''
    prices: 1D array of actual prices (chronological)
    preds: 1D array of predicted next-step prices (aligned to prices)
    Strategy: go long when predicted_return > threshold, else flat (no short).
    Returns: dict with pnl series and simple metrics.
    '''
    prices = np.asarray(prices)
    preds = np.asarray(preds).flatten()
    pred_return = (preds - prices) / (prices + 1e-8)
    positions = (pred_return > threshold).astype(float)  # 1 = long, 0 = flat

    # next-day returns (shifted)
    next_ret = np.zeros_like(prices)
    next_ret[:-1] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

    strategy_ret = positions * next_ret
    # subtract transaction costs on position changes
    trades = np.abs(np.diff(positions, prepend=positions[0]))
    strategy_ret = strategy_ret - trades * pct_cost

    cum_ret = np.cumprod(1 + strategy_ret) - 1
    metrics = {
        'cumulative_return': float(cum_ret[-1]) if len(cum_ret) > 0 else 0.0,
        'annualized_return': float((1 + cum_ret[-1]) ** (252 / len(prices)) - 1) if len(prices) > 0 else 0.0,
        'sharpe_approx': float((np.mean(strategy_ret) / (np.std(strategy_ret) + 1e-9)) * np.sqrt(252))
    }
    return {'returns': strategy_ret, 'cum_ret': cum_ret, 'metrics': metrics}
