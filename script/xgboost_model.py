import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from script.plots import plot_actual_vs_predicted, plot_feature_importance

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data/processed/electricity_clean_long.csv"


# -----------------------------------------

def split_train_test(df, split_date='2017-01-01'):
    """
    Splits the dataset into training and testing based on date.

    Args:
        df (DataFrame): Full daily aggregated dataset.
        split_date (str): Date to split training/testing (default '2017-01-01').

    Returns:
        X_train, X_test, y_train, y_test
    """
    logging.info(f"Splitting train and test with split date = {split_date}...")

    # Make sure 'date' is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Train set: before split_date
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]

    # Features and Target
    feature_cols = [
        'site_id', 'primaryspaceusage', 'sqm', 'yearbuilt', 'lat', 'lng',
        'airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed',
        'meter_reading_lag1', 'meter_reading_lag24', 'meter_reading_roll6',
        'meter_reading_roll12', 'meter_reading_roll24'
    ]

    target_col = 'log_meter_reading'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


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


def save_training_outputs(model, building_id_encoder, predictions_df, feature_level, mae, rmse, r2):
    """
    Save model, encoder, predictions, and experiment results for a feature level.
    """
    models_dir = BASE_DIR / "models"
    results_dir = BASE_DIR / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, models_dir / f"xgboost_model_level{feature_level}.pkl")

    # Save encoder
    joblib.dump(building_id_encoder, models_dir / f"building_id_encoder_level{feature_level}.pkl")

    # Save predictions
    predictions_df.to_csv(results_dir / f"xgboost_predictions_level{feature_level}.csv", index=False)

    # Save experiment results
    experiment_path = results_dir / "xgboost_training_log.csv"
    experiment_entry = pd.DataFrame([{
        "feature_level": feature_level,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }])
    if experiment_path.exists():
        existing_data = pd.read_csv(experiment_path)

        if feature_level in existing_data['feature_level'].values:
            existing_data.loc[existing_data['feature_level'] == feature_level, ['mae', 'rmse', 'r2']] = [mae, rmse, r2]
        else:
            existing_data = pd.concat([existing_data, experiment_entry], ignore_index=True)

        existing_data.to_csv(experiment_path, index=False)
    else:
        experiment_entry.to_csv(experiment_path, index=False)

    logging.info(f"Saved all outputs for Level {feature_level}")


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
    metrics_path = reports_dir / "metrics.csv"
    metrics_df = pd.DataFrame({
        'Model': ['XGBoost'],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"Saved evaluation metrics to {metrics_path}")

    # Plot and save Actual vs Predicted
    plot_actual_vs_predicted(y_test, xgb_preds, model_name="XGBoost", save_path=plots_dir / "xgboost" / "actual_vs_predicted.png")

    # Plot and save Feature importance
    #plot_feature_importance(xgb_model, save_path=plots_dir / "xgboost" / "feature_importance.png")

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
