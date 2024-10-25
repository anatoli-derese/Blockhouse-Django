import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .models import StockData
from sklearn.metrics import mean_absolute_error
import io
import base64
import matplotlib.pyplot as plt

def get_data(symbol, days):
    stock_data_for_specific_days = StockData.objects.filter(symbol=symbol).order_by('-date')[:days]
    df = pd.DataFrame(list(stock_data_for_specific_days.values('date', 'close_price')))
    df = df.sort_values(by='date')
    return df

def train_model(data):
    X = np.arange(len(data)).reshape(-1, 1)
    Y = data['close_price'].values

    model = LinearRegression()
    model.fit(X, Y)

    return model

def predict_prices(symbol, days=30):
    df = get_data(symbol=symbol, days=730) 

    model = train_model(df) 

    future_X = np.arange(len(df), len(df) + days).reshape(-1, 1) 
    predictions = model.predict(future_X) 

    for i in range(days):
        next_pred = predictions[i]
        stock_data, created = StockData.objects.update_or_create(
            symbol=symbol,
            date = pd.Timestamp.now() + pd.Timedelta(days=i+1), 
            defaults={
                'open_price': 0,
                'high_price': 0,
                'low_price': 0,
                'volume': 0,
                'close_price':next_pred,
                'is_predicted': True, 
            }
        )

    return predictions

def calculate_prediction_metrics(actual_prices, predicted_prices):
    mae = mean_absolute_error(actual_prices, predicted_prices)

    return {
        'MAE': mae,
        'Total Predictions': len(predicted_prices)
    }

def generate_stock_price_plot(actual_prices, predicted_prices, dates, symbol):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Prices', color='orange')
    plt.title(f'Actual vs Predicted Prices for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig('stock_price_comparison.png')
    plt.close()

def generate_visualization(symbol, days):
    df = get_data(symbol, days)
    predicted_prices = predict_prices(symbol, days)
    actual_prices = df['close_price'].values
    dates = df['date'].dt.strftime('%Y-%m-%d').values  
    generate_stock_price_plot(actual_prices, predicted_prices, dates, symbol)
    with open('stock_price_comparison.png', "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string  