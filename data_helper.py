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

import os
import glob

import pandas as pd
from datetime import datetime, timedelta


class DataHelper:
    def __init__(self, symbol, timeframe, data_dir="data/btcusd"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_dir = data_dir

    def fetch_historical_data(self):
        file_pattern = os.path.join(self.data_dir, f"{self.symbol}-{self.timeframe}-*.csv")
        csv_files = glob.glob(file_pattern)
        sorted_csv_files = sorted(csv_files)

        df_list = []
        for csv_file in sorted_csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df_list.append(df)
            except Exception as e:
                print(f"Error processing file {csv_file}: {e}")

        return pd.concat(df_list, ignore_index=True)

    def get_halving_date(self, year):
        halving_dates = self.halving_dates()
        for date in halving_dates:
            if date.year == year:
                return date
        raise ValueError(f"No halving date found for the year {year}")

    def halving_dates(self):
        return [datetime(2012, 11, 28), datetime(2016, 7, 9), datetime(2020, 5, 11), datetime(2024, 5, 12),
                datetime(2028, 5, 12)]

    def days_since_last_halving(self, current_date):
        halving_dates = self.halving_dates()
        past_halvings = [date for date in halving_dates if date < current_date]
        if not past_halvings:
            return 0
        last_halving = max(past_halvings)
        return (current_date - last_halving).days

    def generate_future_features(self, data, features, days=90):
        # Get the last known value for each feature
        last_values = data[features].iloc[-1]

        # Calculate historical percentage changes
        pct_changes = data['close'].pct_change().dropna()

        # Focus on percentage changes around all-time highs (e.g., within 5% of the max price)
        all_time_high = data['close'].max()
        threshold = all_time_high * 0.05
        at_high_pct_changes = pct_changes[data['close'] > (all_time_high - threshold)]

        # Calculate the average percentage change around all-time highs
        avg_pct_change_at_high = at_high_pct_changes.mean()

        # Create a DataFrame to hold future feature values
        future_data = pd.DataFrame(
            index=pd.date_range(start=data['timestamp'].iloc[-1] + timedelta(days=1), periods=days))

        # Generate future feature values using the average percentage change
        for feature in features:
            future_values = [last_values[feature]]
            for _ in range(1, days):
                future_values.append(future_values[-1] * (1 + avg_pct_change_at_high))
            future_data[feature] = future_values

        return future_data

    def generate_features_to_halving(self, data, features):
        # Get the last known date and value for each feature
        last_known_date = data['timestamp'].iloc[-1]
        last_values = data[features].iloc[-1]

        # Determine the halving date
        halving_date = self.get_halving_date(2024)  # Replace with actual function to get halving date

        # Calculate the number of days to the halving date
        days_to_halving = (halving_date - last_known_date).days

        # Calculate historical percentage changes
        pct_changes = data['close'].pct_change().dropna()

        # Focus on percentage changes around all-time highs (for example)
        all_time_high = data['close'].max()
        threshold = all_time_high * 0.05
        at_high_pct_changes = pct_changes[data['close'] > (all_time_high - threshold)]

        # Calculate the average percentage change around all-time highs
        avg_pct_change_at_high = at_high_pct_changes.mean()

        # Create a DataFrame to hold future feature values
        future_data = pd.DataFrame(
            index=pd.date_range(start=last_known_date + timedelta(days=1), periods=days_to_halving))

        # Generate future feature values using the average percentage change
        for feature in features:
            future_values = [last_values[feature]]
            for _ in range(1, days_to_halving):
                future_values.append(future_values[-1] * (1 + avg_pct_change_at_high))
            future_data[feature] = future_values

        return future_data
