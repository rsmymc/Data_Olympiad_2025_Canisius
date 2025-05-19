import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from script.utilities.utils import *
from sklearn.preprocessing import StandardScaler

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


def main():
    setup_logging()
    processed_dir, models_dir, reports_dir = prepare_directories()

    # Load and split
    daily_data_path = processed_dir / "electricity_clean_long.csv"
    full_df_daily = pd.read_csv(daily_data_path, parse_dates=['date'])

    X_train, X_test, y_train, y_test = split_train_test(full_df_daily, split_date='2017-01-01')

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = build_tuned_neural_network(input_dim=X_train_scaled.shape[1])

    # Train model
    model, history = train_neural_network(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Predict and Evaluate
    nn_preds = model.predict(X_test_scaled).flatten()
    rmse, mae, r2 = evaluate_model(y_test, nn_preds, model_name="Tuned Neural Network")

    save_evaluation_metrics(rmse, mae, r2, "NeuralNetwork", reports_dir / "nn_metrics.csv")
    save_model(model, models_dir / "neural_network_model.h5")
    save_model(scaler, models_dir / "nn_scaler.pkl")
    save_predictions(y_test, nn_preds, reports_dir / "nn_predictions.csv")


if __name__ == "__main__":
    main()
