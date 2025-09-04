import joblib
import pandas as pd

def load_model(path="models/model.joblib"):
    return joblib.load(path)

def prepare_input(data: dict):
    return pd.DataFrame([data])
