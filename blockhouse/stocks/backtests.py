import pandas as pd
from .models import StockData

def calculate_moving_average(prices, window):
    return prices.rolling(window=window).mean()

def backtest_strategy(symbol, initial_investment, short_window=50, long_window=200):
    # Get data for given symbol
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')

    if not stock_data.exists():
        return {'Status': "error", "Message": "No information available for {symbol}"}
    
    #Convert to Dataframe for easier manipulation
    data = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    data.set_index('date', inplace=True)

    # Calculate Moving Averages
    data['50_MA'] = calculate_moving_average(data['close_price'], short_window)
    data['200_MA'] = calculate_moving_average(data['close_price'], long_window)

    # Initializing relevatn variables for backtesting
    cash = initial_investment
    shares = 0
    portfolio_value = initial_investment
    max_drawdown = 0
    peak_val = initial_investment
    num_trades = 0

    # You want to buy when 50_MA < 200_MA and sell when 50_MA > 200_MA
    for i in range(1, len(data)):
        if data['50_MA'].iloc[i] < data['200_MA'].iloc[i] and shares == 0: # Buy
            shares = (cash) / float(data['close_price'].iloc[i])
            cash = 0
            num_trades += 1
        elif data['50_MA'].iloc[i] > data['200_MA'].iloc[i] and shares > 0: # Sell, given that you have something to sell
            cash = shares * float(data['close_price'].iloc[i])
            shares = 0
            num_trades += 1
        
        # Update portfolio value (total returns) and track max drawdown
        portfolio_value = cash + shares * float(data['close_price'].iloc[i])
        peak_val = max(peak_val, portfolio_value)
        drawdown = (peak_val - portfolio_value) / peak_val
        max_drawdown = max(max_drawdown, drawdown)
    
    # Final value (after all remaining shares are liquidated)
    if shares > 0:
        cash = shares * float(data['close_price'].iloc[-1])
    final_val = cash

    return {
        'Initial Investment': initial_investment,
        'Final Monetary Value ($)': final_val,
        'Total Return': (final_val - initial_investment) / initial_investment * 100,
        'Max Drawdown': max_drawdown * 100,
        'Number of Trades': num_trades
    }

def calculate_metrics(backtest_results):
    """
    Calculate key performance metrics from the backtest
    """
    total_investment = backtest_results['Initial Investment']
    final_value = backtest_results["Final Monetary Value ($)"]
    roi = (final_value - total_investment) / total_investment * 100

    total_trades = backtest_results['Number of Trades']
    
    return {
        'ROI': roi,
        'Total Trades': total_trades,
    }