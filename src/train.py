import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pipeline import create_pipeline
from utils import save_model

df = pd.read_csv("data/energy_data.csv")
target = "kwh"
drop_cols = ["kwh", "date"] 
features = [col for col in df.columns if col not in drop_cols]
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
pipeline = create_pipeline()
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("MSE:", rmse := mean_squared_error(y_test, y_pred) ** 0.5)
print("R2:", r2_score(y_test, y_pred))
save_model(pipeline, "models/model.joblib")
print("✅ Model trained and saved!")
