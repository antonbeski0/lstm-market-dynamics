import numpy as np

def calculate_max_drawdown(cum_returns):
    """
    Calculates the maximum drawdown from a cumulative returns series.
    """
    if len(cum_returns) == 0:
        return 0.0

    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (running_max - cum_returns) / (running_max + 1e-9)
    return np.max(drawdown)

def calculate_performance_metrics(strategy_returns, prices):
    """
    Calculates various performance metrics from strategy returns.
    """
    if len(strategy_returns) == 0 or len(prices) == 0:
        return {
            'cumulative_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }

    cum_returns = np.cumprod(1 + strategy_returns) - 1
    total_return = cum_returns[-1]
    annualized_return = (1 + total_return) ** (252 / len(prices)) - 1
    sharpe_ratio = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)) * np.sqrt(252)

    cum_returns_for_drawdown = np.cumprod(1 + strategy_returns)
    max_drawdown = calculate_max_drawdown(cum_returns_for_drawdown)

    profitable_trades = np.sum(strategy_returns > 0)
    total_trades = np.sum(strategy_returns != 0)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0

    gross_profits = np.sum(strategy_returns[strategy_returns > 0])
    gross_losses = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    metrics = {
        'cumulative_return': float(total_return),
        'annualized_return': float(annualized_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor)
    }
    return metrics

def directional_backtest(prices, predictions, threshold=0.0, transaction_cost_pct=0.0005):
    """
    Performs a directional backtest of a trading strategy.

    Args:
        prices (np.array): 1D array of actual asset prices in chronological order.
        predictions (np.array): 1D array of predicted next-step prices, aligned with `prices`.
        threshold (float): The minimum predicted return to trigger a long position.
        transaction_cost_pct (float): The percentage cost for each transaction.

    Returns:
        dict: A dictionary containing the portfolio's returns, cumulative returns,
              and performance metrics.
    """
    if len(prices) != len(predictions):
        raise ValueError("`prices` and `predictions` must have the same length.")

    prices = np.asarray(prices)
    predictions = np.asarray(predictions).flatten()

    predicted_returns = (predictions - prices) / (prices + 1e-9)
    positions = (predicted_returns > threshold).astype(float)

    asset_returns = np.zeros_like(prices)
    asset_returns[:-1] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-9)

    strategy_returns = positions * asset_returns

    trades = np.abs(np.diff(positions, prepend=positions[0]))
    transaction_costs = trades * transaction_cost_pct
    strategy_returns_after_costs = strategy_returns - transaction_costs

    cumulative_returns = np.cumprod(1 + strategy_returns_after_costs) - 1
    metrics = calculate_performance_metrics(strategy_returns_after_costs, prices)

    return {
        'strategy_returns': strategy_returns_after_costs,
        'cumulative_returns': cumulative_returns,
        'metrics': metrics
    }
