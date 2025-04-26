import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from feature_engineering import apply_feature_level

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FEATURE_LEVEL = 4
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data/processed/electricity_data_long_clean_reduced.csv"

FEATURE_COLUMNS_BY_LEVEL = {
    1: ['hour', 'dayofweek', 'month', 'is_weekend', 'building_id_encoded'],
    2: ['hour', 'dayofweek', 'month', 'is_weekend', 'season_encoded', 'building_id_encoded'],
    3: ['hour', 'dayofweek', 'month', 'is_weekend', 'season_encoded', 'building_id_encoded', 'sqm', 'year_built',
        'primary_space_usage_encoded', 'latitude', 'longitude'],
    4: ['hour', 'dayofweek', 'month', 'is_weekend', 'season_encoded', 'building_id_encoded',
        'sqm', 'year_built', 'primary_space_usage_encoded', 'latitude', 'longitude',
        'airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR',
        'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed']
}

# -----------------------------------------

def encode_building_id(df):
    logging.info("Encoding building_id as numeric labels...")
    le = LabelEncoder()
    df['building_id_encoded'] = le.fit_transform(df['building_id'])
    return df, le

def build_neural_network(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_model(df, feature_level):
    feature_cols = FEATURE_COLUMNS_BY_LEVEL.get(feature_level)
    feature_cols = [col for col in feature_cols if col in df.columns]
    X = df[feature_cols]
    y = df['consumption_kwh']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    model = build_neural_network(input_dim=X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    logging.info("Evaluating Neural Network model...")
    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Feature Level {FEATURE_LEVEL} - MAE: {mae:.4f}")
    logging.info(f"Feature Level {FEATURE_LEVEL} - RMSE: {rmse:.4f}")
    logging.info(f"Feature Level {FEATURE_LEVEL} - R2 Score: {r2:.4f}")

    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    })

    return model, scaler, mae, rmse, r2, predictions_df

def save_training_outputs(model, building_id_encoder, scaler, predictions_df, feature_level, mae, rmse, r2):
    models_dir = BASE_DIR / "models"
    results_dir = BASE_DIR / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model.save(models_dir / f"neural_network_model_level{feature_level}.h5")
    joblib.dump(building_id_encoder, models_dir / f"building_id_encoder_level{feature_level}.pkl")
    joblib.dump(scaler, models_dir / f"scaler_level{feature_level}.pkl")
    predictions_df.to_csv(results_dir / f"neural_network_predictions_level{feature_level}.csv", index=False)

    experiment_path = results_dir / "neural_network_training_log.csv"
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
    logging.info(f"Loading electricity data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=['timestamp'])

    df = apply_feature_level(df, level=FEATURE_LEVEL)
    df, building_id_encoder = encode_building_id(df)

    model, scaler, mae, rmse, r2, predictions_df = train_model(df, FEATURE_LEVEL)

    save_training_outputs(model, building_id_encoder, scaler, predictions_df, FEATURE_LEVEL, mae, rmse, r2)

if __name__ == "__main__":
    main()
