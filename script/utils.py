import logging
import pandas as pd
import numpy as np


def load_predictions(reports_dir, filename):
    preds_path = reports_dir / filename
    if not preds_path.exists():
        logging.error(f"Predictions file {filename} not found!")
        exit()
    preds_df = pd.read_csv(preds_path)
    preds_df['y_true'] = np.log1p(preds_df['y_true'])
    preds_df['y_pred'] = np.log1p(preds_df['y_pred'])
    return preds_df


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
