import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['account_no', 'date'])

    # Time-based features 
    df['dayofweek'] = df['date'].dt.dayofweek    
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['day'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Cyclical encoding (sin/cos) 
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

    # Lag features for kWh 
    for lag in range(1, 8):  # lag_1 ... lag_7
        df[f'lag_{lag}'] = df.groupby('account_no')['kwh'].shift(lag)

    # Rolling mean features for kWh 
    df['rolling_mean_7'] = df.groupby('account_no')['kwh'].shift(1).rolling(window=7).mean()
    df['rolling_mean_30'] = df.groupby('account_no')['kwh'].shift(1).rolling(window=30).mean()

    # Temperature lag features 
    for lag in range(1, 4):  # temp_lag_1 ... temp_lag_3
        df[f'temp_lag_{lag}'] = df.groupby('account_no')['avg_temp'].shift(lag)

    # Temperature rolling mean 
    df['temp_roll_mean_3'] = df.groupby('account_no')['avg_temp'].shift(1).rolling(window=3).mean()
    df['temp_roll_mean_7'] = df.groupby('account_no')['avg_temp'].shift(1).rolling(window=7).mean()

    # Interaction feature
    df['temp_kwh_interaction'] = df['kwh'] * df['avg_temp']
    return df


def get_expected_features() -> list:
    return [
        'account_no','kwh','avg_temp','dayofweek','quarter','month','year',
        'dayofyear','day','weekofyear','is_weekend','dayofyear_sin','dayofyear_cos',
        'weekofyear_sin','weekofyear_cos','lag_1','lag_2','lag_3','lag_4',
        'lag_5','lag_6','lag_7','rolling_mean_7','rolling_mean_30',
        'temp_lag_1','temp_lag_2','temp_lag_3','temp_roll_mean_3',
        'temp_roll_mean_7','temp_kwh_interaction','date'
    ]
