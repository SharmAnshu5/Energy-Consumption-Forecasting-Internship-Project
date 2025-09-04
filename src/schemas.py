# src/schemas.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class FeatureRow(BaseModel):
    account_no: int
    avg_temp: float
    dayofweek: int
    quarter: int
    month: int
    year: int
    dayofyear: int
    day: int
    weekofyear: int
    is_weekend: int
    dayofyear_sin: float
    dayofyear_cos: float
    weekofyear_sin: float
    weekofyear_cos: float
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_5: float
    lag_6: float
    lag_7: float
    rolling_mean_7: float
    rolling_mean_30: float
    temp_lag_1: float
    temp_lag_2: float
    temp_lag_3: float
    temp_roll_mean_7: float
    temp_roll_mean_30: float
    temp_kwh_intera: float

class PredictFeaturesRequest(BaseModel):
    rows: List[FeatureRow]

class PredictResponseRow(BaseModel):
    account_no: int
    predicted_kwh: float

class PredictFeaturesResponse(BaseModel):
    predictions: List[PredictResponseRow]

class HistoryRow(BaseModel):
    date: str                
    kwh: float               
    avg_temp: float

class PredictNextRequest(BaseModel):
    account_no: int
    history: List[HistoryRow] 
    next_date: str            
    next_avg_temp: float      

class PredictNextResponse(BaseModel):
    account_no: int
    next_date: str
    predicted_kwh: float
