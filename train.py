import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("dataset/sales.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["product", "date"]).reset_index(drop=True)

group = df.groupby("product")["sales"]
df["lag_1"] = group.shift(1)
df["lag_7"] = group.shift(7)
df["rolling_7"] = group.shift(1).rolling(7).mean()
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df = df.dropna().reset_index(drop=True)

features = ["lag_1", "lag_7", "rolling_7", "day_of_week", "month"]
cut = int(len(df) * 0.80)
train = df.iloc[:cut]
test = df.iloc[cut:]

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(train[features], train["sales"])

pred = model.predict(test[features])
mae = mean_absolute_error(test["sales"], pred)
rmse = np.sqrt(mean_squared_error(test["sales"], pred))

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

joblib.dump({"model": model, "features": features}, "demand_model.joblib")

pd.DataFrame({"Metric":["MAE","RMSE"],"Value":[mae,rmse]}).to_csv(
    "model_evaluation_report.csv", index=False
)
out = test[["date","product","sales"]].copy()
out["predicted_sales"] = pred
out["absolute_error"] = (out["sales"]-out["predicted_sales"]).abs()
out.to_csv("demand_predictions.csv", index=False)
print("Model saved as demand_model.joblib")
