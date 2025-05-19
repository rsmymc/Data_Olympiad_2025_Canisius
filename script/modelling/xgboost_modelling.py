import pandas as pd
import xgboost as xgb
import logging
from pathlib import Path
from script.utilities.utils import *


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


def main():
    setup_logging()
    processed_dir, models_dir, reports_dir = prepare_directories()

    # Load and split
    daily_data_path = processed_dir / "electricity_clean_long.csv"
    full_df_daily = pd.read_csv(daily_data_path, parse_dates=['date'])

    X_train, X_test, y_train, y_test = split_train_test(full_df_daily, split_date='2017-01-01')

    # Train model
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    # Predict and Evaluate
    xgb_preds = xgb_model.predict(X_test)
    rmse, mae, r2 = evaluate_model(y_test, xgb_preds, model_name="XGBoost")

    save_evaluation_metrics(rmse, mae, r2, "XGBoost", reports_dir / "xgboost_metrics.csv")
    save_model(xgb_model, models_dir / "xgboost_model.pkl")
    save_predictions(y_test, xgb_preds, reports_dir / "xgboost_predictions.csv")


if __name__ == "__main__":
    main()
