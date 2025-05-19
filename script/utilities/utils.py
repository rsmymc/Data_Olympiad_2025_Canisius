import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def prepare_directories():
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_dir = base_dir / "data/processed"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, models_dir, reports_dir



def split_train_test(df, feature_cols=None, target_col='log_meter_reading', split_date='2017-01-01'):
    """
    Splits the dataset into training and testing based on date.

    Args:
        df (DataFrame): Full dataset with 'date' column.
        feature_cols (list, optional): List of feature columns.
        target_col (str): Name of the target column.
        split_date (str): Date string to split train/test.

    Returns:
        X_train, X_test, y_train, y_test
    """
    logging.info(f"Splitting dataset with split date: {split_date}")

    df['date'] = pd.to_datetime(df['date'])

    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]

    if feature_cols is None:
        feature_cols = [
            'site_id', 'primaryspaceusage', 'sqm', 'yearbuilt', 'lat', 'lng',
            'airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed',
            'meter_reading_lag1', 'meter_reading_lag24', 'meter_reading_roll6',
            'meter_reading_roll12', 'meter_reading_roll24'
        ]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    logging.info("\n Data Shapes:")
    logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


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
    joblib.dump(model, save_path)


def save_evaluation_metrics(rmse, mae, r2, model_name, save_path):
    logging.info(f"Saving model to {save_path}...")
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })
    metrics_df.to_csv(save_path, index=False)
    logging.info(f"Saved evaluation metrics to {save_path}")


def save_predictions(y_test, y_preds, save_path):
    preds_df = pd.DataFrame({
        'y_true': np.expm1(y_test),
        'y_pred': np.expm1(y_preds)
    })
    preds_df.to_csv(save_path, index=False)
    logging.info(f"Saved predictions to {save_path}")


def load_predictions(reports_dir, filename):
    preds_path = reports_dir / filename
    if not preds_path.exists():
        logging.error(f"Predictions file {filename} not found!")
        exit()
    preds_df = pd.read_csv(preds_path)
    preds_df['y_true'] = np.log1p(preds_df['y_true'])
    preds_df['y_pred'] = np.log1p(preds_df['y_pred'])
    return preds_df
