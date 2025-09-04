import pandas as pd
import numpy as np

def prepare_features(df):
    if "weekday" in df.columns:
        df = df.rename(columns={"weekday": "dayofweek"})
    expected_features = [
        'account_no','avg_temp','dayofweek','quarter','month','year','dayofyear','day',
        'weekofyear','is_weekend','dayofyear_sin','dayofyear_cos','weekofyear_sin',
        'weekofyear_cos','lag_1','lag_2','lag_3','lag_4','lag_5','lag_6','lag_7',
        'rolling_mean_7','rolling_mean_30','temp_lag_1','temp_lag_2','temp_lag_3',
        'temp_roll_mean_3','temp_roll_mean_7','temp_kwh_interaction'
    ]    
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  
    df = df[expected_features]
    return df


