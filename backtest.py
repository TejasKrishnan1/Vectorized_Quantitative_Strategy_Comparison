#comparing mean reversion, factor modeling momentum, buy-hold for the SPY
# 
# #main imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#for data
import yfinance as yf

#plotting config
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 7)


#fetch data
ticker = 'SPY'
df = yf.download(ticker, start='2021-01-01', end='2026-01-01')

#to fix the multi index columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

#get close prices
df = df[['Close']].copy()

#inspect data
print(df.head())
df.info()


window = 20 #arbitrary rolling window
 
#we use .rolling rather than a loop
df['SMA'] = df['Close'].rolling(window=window).mean()
df['Rolling_Std'] = df['Close'].rolling(window=window).std()
 
#calculate z score (close - mean) / std
df['Z_Score'] = (df['Close'] - df['SMA']) / df['Rolling_Std']
 
#drop garbage values from the rolling window warm-up period
df.dropna(inplace=True)
 
#plot the mean reversion indicator (20-day SMA)
df[['Close', 'SMA']].plot(title=f'{ticker} Price vs {window}-Day SMA (Mean Reversion)')
plt.show()
 
 
#np.where(condition, value_if_true, value_if_false) is good for making masks
#we have (1, -1, 0) for (Long, Short, Flat)
df['MR_Signal'] = np.where(df['Z_Score'] < -2, 1,
                   np.where(df['Z_Score'] > 2, -1, 0))
 
#shift the signal to create the trade (avoid look-ahead bias)
#normally we forward fill to hold until next signal
df['MR_Position'] = df['MR_Signal'].shift(1).ffill()
 
print(df['MR_Position'].value_counts())
 
#drop the first row(s) that are NaN because of the shift(1) above
df.dropna(inplace=True)
 
 
#calculate the market returns (we do this with log returns)
market_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0).values
 
 
def backtest_strategy(positions, market_returns):
    """
    Takes a NumPy array of positions (+1, -1, 0) and a NumPy array of market returns,
    and returns (strategy_returns, cumulative_returns) for that strategy.
    """
    #we can very quickly backtest by just multiplying our positions by the market returns
    strategy_returns = positions * market_returns
 
    #then we can just sum up all those little movements and exponentiate to get back the value
    cumulative_returns = np.exp(np.cumsum(strategy_returns)) - 1
 
    return strategy_returns, cumulative_returns
 
 
#get our positions from earlier, one NumPy array per strategy
mean_reversion_positions = df['MR_Position'].values
 
#run the backtest
mean_reversion_returns, cum_mean_reversion_returns = backtest_strategy(mean_reversion_positions, market_returns)
 
 
#we assume 252 trading days per year to annualize our metrics
TRADING_DAYS_PER_YEAR = 252
 
 
def calculate_performance_metrics(strategy_returns, cumulative_returns):
    """
    Takes a strategy's daily returns and cumulative returns, and returns a dictionary of
    Annual Return, Annual Volatility, Sharpe Ratio, and Max Drawdown.
    """
    annual_return = np.mean(strategy_returns) * TRADING_DAYS_PER_YEAR
    annual_volatility = np.std(strategy_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(TRADING_DAYS_PER_YEAR)
 
    #max drawdown: the biggest drop from a running peak in the equity curve
    equity_curve = 1 + cumulative_returns
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_peak) / running_peak
    max_drawdown = np.min(drawdown)
 
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
 
 
mean_reversion_metrics = calculate_performance_metrics(mean_reversion_returns, cum_mean_reversion_returns)
 
performance_summary = pd.DataFrame({
    'Mean Reversion': mean_reversion_metrics
}).T
 
print(performance_summary)
 
 
plt.figure(figsize=(14, 7))
plt.plot(df.index, cum_mean_reversion_returns, label='Mean Reversion Strategy', color='cyan', linewidth=2)
plt.title('Mean Reversion Strategy: Equity Curve')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
