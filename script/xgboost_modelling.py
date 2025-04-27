import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import split_train_test


def train_xgboost(X_train, y_train, X_val, y_val):
    logging.info("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=10
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return model


def evaluate_model(y_true, y_pred, model_name="Model"):
    logging.info(f"Evaluating {model_name}...")
    y_true_inv = np.expm1(y_true)
    y_pred_inv = np.expm1(y_pred)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)
    logging.info(f"{model_name} RMSE: {rmse:.2f}")
    logging.info(f"{model_name} MAE: {mae:.2f}")
    logging.info(f"{model_name} R²: {r2:.4f}")
    return rmse, mae, r2


def save_model(model, save_path):
    logging.info(f"Saving model to {save_path}...")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data/processed"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    plots_dir = base_dir / "plots"

    daily_data_path = processed_dir / "electricity_clean_long.csv"
    full_df_daily = pd.read_csv(daily_data_path, parse_dates=['date'])

    X_train, X_test, y_train, y_test = split_train_test(full_df_daily, split_date='2017-01-01')

    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    xgb_preds = xgb_model.predict(X_test)

    rmse, mae, r2 = evaluate_model(y_test, xgb_preds, model_name="XGBoost")

    # Save model
    model_path = models_dir / "xgboost_model.pkl"
    save_model(xgb_model, model_path)

    # Save evaluation metrics
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "xgboost_metrics.csv"
    metrics_df = pd.DataFrame({
        'Model': ['XGBoost'],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"Saved evaluation metrics to {metrics_path}")

    # Save predictions and true values for feature analysis
    preds_df = pd.DataFrame({
        'y_true': np.expm1(y_test),
        'y_pred': np.expm1(xgb_preds)
    })
    preds_path = reports_dir / "xgboost_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logging.info(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()
