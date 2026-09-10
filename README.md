# Supply Chain Demand Forecasting

Machine learning baseline for forecasting future product demand from historical sales.

## Workflow
Historical sales → date processing → lag/rolling feature engineering → chronological split → Random Forest → MAE/RMSE → saved model → Streamlit dashboard.

## Dataset
`dataset/sales.csv` is a **simulated educational dataset** because the uploaded guide states that its actual `sales.csv` is not included. Replace it with a properly licensed real dataset for final experiments.

Columns: `date`, `product`, `sales`.

## Features
- lag_1
- lag_7
- rolling_7
- day_of_week
- month

Historical values are shifted to avoid time-series leakage.

## Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Training creates `demand_model.joblib`, `model_evaluation_report.csv`, and `demand_predictions.csv`.

The guide's 15–20% WMAPE improvement is a target for an advanced system, not a measured result. Metrics are calculated only when the code is run.

## Future scope
LSTM, XGBoost, Temporal Fusion Transformer (TFT), SHAP, promotions, holidays, inventory, competitor prices, web trends, macroeconomic variables, automated retraining, drift detection and MLOps/AWS deployment.

## Disclaimer
This is an educational forecasting prototype. Forecasts are planning aids and should be compared with actual demand and business constraints.

AUTHOR 
ANU SHREE KA
