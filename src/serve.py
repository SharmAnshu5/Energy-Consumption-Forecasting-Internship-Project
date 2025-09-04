from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from src.pipeline import prepare_features  
model = joblib.load("models/model.joblib")
base_df = pd.read_csv("data/energy_data.csv")
if "date" not in base_df.columns:
    raise RuntimeError("Expected a 'date' column in data/energy_data.csv")

base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce", dayfirst=True)
if base_df["date"].isna().any():
    base_df = base_df.dropna(subset=["date"]).copy()
app = FastAPI()

class InputData(BaseModel):
    account_no: int
    date: str 

def build_feature_row_for_request(account_no: int, date_str: str) -> pd.DataFrame:    
    try:
        req_date = pd.to_datetime(date_str, errors="raise", dayfirst=True)
    except Exception:
        req_date = pd.to_datetime(date_str, errors="raise", format="ISO8601")
    acc_hist = base_df[base_df["account_no"] == account_no].copy()
    if acc_hist.empty:
        raise HTTPException(status_code=404, detail=f"No history found for account {account_no}.")
    acc_hist = acc_hist.sort_values("date")
    placeholder = {
        "account_no": account_no,
        "date": req_date
    }
    acc_plus_one = pd.concat([acc_hist, pd.DataFrame([placeholder])], ignore_index=True)
    feats_all = prepare_features(acc_plus_one)
    feats_row = feats_all.iloc[[-1]].copy()
    if "kwh" in feats_row.columns:
        feats_row = feats_row.drop(columns=["kwh"])

    return feats_row

def align_to_model_features(X_row: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        return X_row.reindex(columns=expected, fill_value=0)
    try:
        preproc = model.named_steps.get("preprocessor")
        if preproc is not None and hasattr(preproc, "get_feature_names_out"):
            expected = list(preproc.get_feature_names_out())
            return X_row.reindex(columns=expected, fill_value=0)
    except Exception:
        pass
    return X_row
@app.post("/predict")
def predict(request: InputData):
    try:
        X_row = build_feature_row_for_request(request.account_no, request.date)
        X_row = align_to_model_features(X_row)
        for col in X_row.columns:
            if not pd.api.types.is_numeric_dtype(X_row[col]):
                X_row[col] = pd.to_numeric(X_row[col], errors="coerce").fillna(0)
        yhat = model.predict(X_row)
        pred = float(yhat[0])
        return {
            "account_no": request.account_no,
            "date": request.date,
            "predicted_kwh": pred
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
