"""
# BITCOIN PRICE PREDICTION AND ANALYSIS BY SIMULATING CONSISTENT GROWTH NEAR ALL-TIME HIGH

## Author: Iman Samizadeh
## Contact: Iman.samizadeh@gmail.com
## License: MIT License (See below)

MIT License

Copyright (c) 2024 Iman Samizadeh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Disclaimer

This code and its predictions are for educational purposes only and should not be considered as financial or investment advice.
The author and anyone associated with the code is not responsible for any financial losses or decisions based on the code's output.

"""

import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from data_helper import DataHelper
from technical_analysis import TechnicalAnalysis

np.random.seed(0)

"""
## Fetching and Preparing Data

Fetch historical Bitcoin price data, convert timestamps, and calculate volatility.
"""

# Initialize DataHelper with BTC/USD daily data
data = DataHelper('btcusd', 'd1')
btc_data = data.fetch_historical_data()

# Convert timestamp to datetime format
btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'], unit='ms')

# Check if 'high' and 'low' columns exist for volatility calculation
if 'high' in btc_data.columns and 'low' in btc_data.columns:
    btc_data['volatility'] = btc_data['high'] - btc_data['low']
else:
    raise ValueError("Columns 'high' and 'low' are required for 'volatility' calculation")

# Fetch Bitcoin halving dates and calculate days since the last halving
halving_dates = data.halving_dates()
btc_data['days_since_halving'] = btc_data['timestamp'].apply(lambda x: data.days_since_last_halving(x))

"""
## Estimating Future Bitcoin Prices

Predict future Bitcoin prices by simulating consistent growth near all-time highs.
"""

# Get the last date from the data
last_date = btc_data['timestamp'].iloc[-1]

# Generate future dates (5 years)
future_dates = [last_date + timedelta(days=i) for i in range(1, 365 * 7 + 1)]

# Calculate all-time high price and define a threshold for 'near all-time high' periods
max_historical_price = btc_data['close'].max()
threshold = max_historical_price * 0.05  # within 5% of the all-time high

# Filter data to include periods near the all-time high
near_ath_data = btc_data[btc_data['close'] >= (max_historical_price - threshold)]

# Calculate the average volatility during these near all-time high periods
ath_average_volatility = near_ath_data['volatility'].mean()

# Assume a small positive increment for consistent growth
increment = ath_average_volatility * 0.1  # Adjust this factor to control the optimism level

# Apply the increment to estimate future prices
estimated_future_prices = [max_historical_price]

for _ in range(1, 365 * 7):
    # Consistently add the small increment
    estimated_future_prices.append(estimated_future_prices[-1] + increment)

estimated_future_prices = pd.Series(estimated_future_prices,
                                    index=pd.date_range(start=btc_data['timestamp'].iloc[-1], periods=365 * 7,
                                                        freq='D'))
last_price = estimated_future_prices.iloc[-1]

"""
## Technical Analysis

Perform technical analysis and calculate indicators.
"""

# Calculate the 7-day moving average of the opening price
btc_data['open_ma_7'] = btc_data['open'].rolling(window=7).mean()

# Calculate the Relative Strength Index (RSI)
btc_data['rsi'] = TechnicalAnalysis().relative_strength_idx(btc_data)

# Create lagged close price features
for lag in [1, 3, 7, 14, 30]:
    btc_data[f'lagged_close_{lag}'] = btc_data['close'].shift(lag)

# Calculate rolling mean and standard deviation for different windows
for window in [7, 14, 30]:
    btc_data[f'rolling_mean_{window}'] = btc_data['close'].rolling(window=window).mean()
    btc_data[f'rolling_std_{window}'] = btc_data['close'].rolling(window=window).std()

# Drop rows with missing values and reset the index
btc_data = btc_data.dropna().reset_index(drop=True)

"""
## Plotting Bitcoin Price Data

Visualize Bitcoin price data and estimated future prices.
"""

# Function to format price labels for the y-axis
def human_friendly_dollar(x, pos):
    if x >= 1e6:
        return '${:1.1f}M'.format(x * 1e-6)
    elif x >= 1e3:
        return '${:1.0f}K'.format(x * 1e-3)
    return '${:1.0f}'.format(x)


# Set plot style and size
plt.style.use('dark_background')
plt.figure(figsize=(20, 10))

# Plot actual Bitcoin prices
plt.plot(btc_data['timestamp'], btc_data['close'], label='Actual Prices', color='cyan', linewidth=1)

# Plot estimated future top prices
plt.plot(future_dates, estimated_future_prices, label='Estimated Future Top Prices', color='orange', linestyle='--',
         linewidth=2)

# Annotate the last estimated price
plt.annotate(f'${last_price:,.2f}',  # Format the price as a string
             xy=(last_date, last_price),
             xytext=(last_date + timedelta(days=10), last_price),
             arrowprops=dict(facecolor='white', arrowstyle='->'),
             fontsize=12, color='white')

# Annotate prices at the start of each new year
prev_year = future_dates[0].year

for i in range(len(future_dates)):
    current_year = future_dates[i].year
    if current_year != prev_year:
        # Annotate the price at the start of the new year
        plt.annotate(f'{current_year}\n${estimated_future_prices[i]:,.2f}',
                     xy=(future_dates[i], estimated_future_prices[i]),
                     xytext=(future_dates[i] + timedelta(days=30), estimated_future_prices[i]),
                     arrowprops=dict(facecolor='white', arrowstyle='->'),
                     fontsize=10, color='white',
                     horizontalalignment='right')
        prev_year = current_year

# Add vertical lines for Bitcoin halving dates
for halving_date in halving_dates:
    plt.axvline(x=halving_date, color='red', linestyle='--', linewidth=2)
    plt.annotate(f'Halving {halving_date.strftime("%Y-%m-%d")}',
                 xy=(halving_date, plt.ylim()[1]),
                 xytext=(halving_date, plt.ylim()[1] * 0.6),
                 arrowprops=dict(facecolor='white', arrowstyle='->', connectionstyle='arc3,rad=-0.2'),
                 fontsize=12, color='white', horizontalalignment='right')

# Annotate the current Bitcoin price
current_price = btc_data['close'].iloc[-1]
current_date = btc_data['timestamp'].iloc[-1]
plt.annotate(f'Current Price: ${current_price:,.2f}',
             xy=(current_date, current_price),
             xytext=(current_date + timedelta(days=150), current_price),
             arrowprops=dict(facecolor='white', arrowstyle='->'),
             fontsize=12, color='white')

# Customize the plot axis labels and legend
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(human_friendly_dollar))
plt.gcf().autofmt_xdate()

plt.title('BITCOIN PRICE PREDICTION AND ANALYSIS BY SIMULATING CONSISTENT GROWTH NEAR ALL-TIME HIGHS', fontsize=20,
          color='yellow')
plt.xlabel('Date', fontsize=16, color='white')
plt.ylabel('BTC Price (USD)', fontsize=16, color='white')
plt.legend(loc='upper left', fontsize=14)

plt.show()
