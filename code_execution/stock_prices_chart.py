# filename: stock_prices_chart.py
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker symbols for META and Apple
meta_symbol = 'META'
apple_symbol = 'AAPL'

# Download historical stock price data from 2022 to 2024 for META and Apple
meta_data = yf.download(meta_symbol, start='2022-01-01', end='2024-01-01')
apple_data = yf.download(apple_symbol, start='2022-01-01', end='2024-01-01')

# Drop any rows with missing or NaN values
meta_data.dropna(inplace=True)
apple_data.dropna(inplace=True)

# Plot the Close prices for META and Apple
plt.figure(figsize=(14, 7))
plt.plot(meta_data['Close'], label='META')
plt.plot(apple_data['Close'], label='Apple')
plt.title('META and Apple Stock Prices from 2022 to 2024')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()