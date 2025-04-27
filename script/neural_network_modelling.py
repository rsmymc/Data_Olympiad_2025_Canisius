import logging
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras import layers, models, callbacks
from utils import split_train_test
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --- Build Neural Network ---
def build_tuned_neural_network(input_dim):
    """
    Builds a tuned dense neural network for regression tasks.

    Args:
        input_dim (int): Number of input features.

    Returns:
        model (Sequential): Compiled Keras model.
    """
    logging.info("Building Tuned Neural Network...")

    model = models.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse'
    )

    return model


# --- Train Neural Network ---
def train_neural_network(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=2048):
    """
    Trains the neural network with early stopping.

    Args:
        model (Sequential): Compiled Keras model.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.

    Returns:
        model (Sequential): Trained Keras model.
        history: Training history object.
    """
    logging.info("Training Neural Network...")

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


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

    # Load and split
    daily_data_path = processed_dir / "electricity_clean_long.csv"
    full_df_daily = pd.read_csv(daily_data_path, parse_dates=['date'])

    X_train, X_test, y_train, y_test = split_train_test(full_df_daily, split_date='2017-01-01')

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    scaler_path = models_dir / "nn_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved scaler to {scaler_path}")

    # Build model
    model = build_tuned_neural_network(input_dim=X_train_scaled.shape[1])

    # Train model
    model, history = train_neural_network(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Predict and Evaluate
    nn_preds = model.predict(X_test_scaled).flatten()
    rmse, mae, r2 = evaluate_model(y_test, nn_preds, model_name="Tuned Neural Network")

    # Save model
    model_path = models_dir / "neural_network_model.h5"
    save_model(model, model_path)

    # Save evaluation metrics
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "nn_metrics.csv"
    metrics_df = pd.DataFrame({
        'Model': ['NeuralNetwork'],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"Saved evaluation metrics to {metrics_path}")

    # Save predictions and true values for feature analysis
    preds_df = pd.DataFrame({
        'y_true': np.expm1(y_test),
        'y_pred': np.expm1(nn_preds)
    })
    preds_path = reports_dir / "nn_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logging.info(f"Saved predictions to {preds_path}")

if __name__ == "__main__":
    main()
