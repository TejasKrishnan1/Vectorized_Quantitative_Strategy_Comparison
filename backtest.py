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

#momentum assumes the opposite idea of mean reversion -> if price is trending, it will keep trending
momentum_window = 50 #longer window so we capture the trend, not the daily noise

#same rolling logic as the mean reversion SMA above, just a longer lookback
df['Momentum_SMA'] = df['Close'].rolling(window=momentum_window).mean()
 
#drop garbage values from the rolling window warm-up period
df.dropna(inplace=True)
 
#plot the mean reversion indicator (20-day SMA)
df[['Close', 'SMA']].plot(title=f'{ticker} Price vs {window}-Day SMA (Mean Reversion)')
plt.show()

#plot the momentum indicator (50-day SMA)
df[['Close', 'Momentum_SMA']].plot(title=f'{ticker} Price vs {momentum_window}-Day SMA (Momentum)')
plt.show()
 
 
#if price is more than 2 standard deviations below the mean (oversold), go long (+1), betting it reverts up. If more than 2 above (overbought), go short (-1), betting it reverts down. Otherwise stay flat (0)
df['MR_Signal'] = np.where(df['Z_Score'] < -2, 1,
                   np.where(df['Z_Score'] > 2, -1, 0))
 
#shift the signal to create the trade (avoid look-ahead bias)
#normally we forward fill to hold until next signal
df['MR_Position'] = df['MR_Signal'].shift(1).ffill()
 
print(df['MR_Position'].value_counts())

#momentum trading logic: go long (+1) if price is above the 50-day SMA (uptrend), go short (-1) if below (downtrend)
df['Momentum_Signal'] = np.where(df['Close'] > df['Momentum_SMA'], 1, -1)

#same look-ahead bias fix as mean reversion
df['Momentum_Position'] = df['Momentum_Signal'].shift(1).ffill()

print(df['Momentum_Position'].value_counts())

#buy & hold benchmark: always long, no signal needed, just stay at +1 the whole time
df['BuyHold_Position'] = 1
 
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
momentum_positions = df['Momentum_Position'].values
buy_hold_positions = df['BuyHold_Position'].values
 
#run the backtest for all three strategies, same function, same market_returns
mean_reversion_returns, cum_mean_reversion_returns = backtest_strategy(mean_reversion_positions, market_returns)
momentum_returns, cum_momentum_returns = backtest_strategy(momentum_positions, market_returns)
buy_hold_returns, cum_buy_hold_returns = backtest_strategy(buy_hold_positions, market_returns)
 
 
#we assume 252 trading days per year to annualize our metrics
TRADING_DAYS_PER_YEAR = 252
 
 
def calculate_performance_metrics(strategy_returns, cumulative_returns):
    """
    Takes a strategy's daily returns and cumulative returns, and returns a dictionary of
    Annual Return, Annual Volatility, Sharpe Ratio, and Max Drawdown.
    """
    annual_return = np.mean(strategy_returns) * TRADING_DAYS_PER_YEAR  #average daily return scaled up to 252 trading days
    annual_volatility = np.std(strategy_returns) * np.sqrt(TRADING_DAYS_PER_YEAR) #daily standard devation by sqrt(252) to annualize.
    sharpe_ratio = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(TRADING_DAYS_PER_YEAR) # risk-adjusted return. Mean return divided by volatility, annualized the same way
 
    #max drawdown: =tracks the running peak of the equity curve and measures the worst percentage drop from any peak to a subsequent trough
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
momentum_metrics = calculate_performance_metrics(momentum_returns, cum_momentum_returns)
buy_hold_metrics = calculate_performance_metrics(buy_hold_returns, cum_buy_hold_returns)
 
performance_summary = pd.DataFrame({
    'Mean Reversion': mean_reversion_metrics,
    'Momentum': momentum_metrics,
    'Buy & Hold': buy_hold_metrics
}).T

#sort so the best risk-adjusted strategy shows up first
performance_summary = performance_summary.sort_values(by='Sharpe Ratio', ascending=False)
 
print(performance_summary)
 
 
plt.figure(figsize=(14, 7))
plt.plot(df.index, cum_buy_hold_returns, label='Buy & Hold SPY Benchmark', color='gray', linewidth=2)
plt.plot(df.index, cum_mean_reversion_returns, label='Mean Reversion Strategy', color='cyan', linewidth=2)
plt.plot(df.index, cum_momentum_returns, label='Momentum Strategy', color='orange', linewidth=2)
plt.title('Strategy Comparison: Equity Curves')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()