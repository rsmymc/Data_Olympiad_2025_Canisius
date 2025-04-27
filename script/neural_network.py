import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, callbacks
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from script.plots import plot_actual_vs_predicted


def split_train_test(df, feature_cols, target_col='log_meter_reading', split_date='2017-01-01'):
    logging.info(f"Splitting dataset based on date: {split_date}")
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test

def build_model(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
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
    logging.info(f"{model_name} R2: {r2:.4f}")
    return rmse, mae, r2

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data/processed"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    plots_dir = base_dir / "plots"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    daily_data_path = processed_dir / "electricity_clean_long.csv"
    full_df_daily = pd.read_csv(daily_data_path, parse_dates=['date'])

    feature_cols = [
        'site_id', 'primaryspaceusage', 'sqm', 'yearbuilt', 'lat', 'lng',
        'airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed',
        'meter_reading_lag1', 'meter_reading_lag24', 'meter_reading_roll6',
        'meter_reading_roll12', 'meter_reading_roll24'
    ]

    X_train, X_test, y_train, y_test = split_train_test(full_df_daily, feature_cols)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    scaler_path = models_dir / "nn_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved scaler to {scaler_path}")

    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, batch_size=32)

    # Predict
    y_pred = model.predict(X_test_scaled).flatten()

    # Evaluate
    rmse, mae, r2 = evaluate_model(y_test, y_pred, model_name="NeuralNetwork")

    # Save model
    model_path = models_dir / "neural_network_model.h5"
    model.save(model_path)
    logging.info(f"Saved neural network model to {model_path}")

    # Save evaluation metrics
    metrics_path = reports_dir / "nn_metrics.csv"
    metrics_df = pd.DataFrame({
        'Model': ['NeuralNetwork'],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"Saved neural network metrics to {metrics_path}")

    # Plot and save Actual vs Predicted
    plot_actual_vs_predicted(y_test, y_pred, model_name="Neural Network",
                             save_path=plots_dir / "neural_network" / "actual_vs_predicted.png")

    # Save predictions
    preds_df = pd.DataFrame({
        'y_true': np.expm1(y_test),
        'y_pred': np.expm1(y_pred)
    })
    preds_path = reports_dir / "nn_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logging.info(f"Saved neural network predictions to {preds_path}")
