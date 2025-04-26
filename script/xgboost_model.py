import pandas as pd
import xgboost as xgb
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from feature_engineering import apply_feature_level
import numpy as np

# ---------------- CONFIG ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FEATURE_LEVEL = 1
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data/processed/electricity_data_long_clean.csv"

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
    """
    Encode building_id to numeric labels.
    """
    logging.info("Encoding building_id as numeric labels...")
    le = LabelEncoder()
    df['building_id_encoded'] = le.fit_transform(df['building_id'])
    return df, le


def train_xgboost(df, feature_level):
    """
    Train XGBoost model on the provided DataFrame.
    """
    logging.info("Preparing training data...")

    feature_cols = FEATURE_COLUMNS_BY_LEVEL.get(feature_level)
    feature_cols = [col for col in feature_cols if col in df.columns]  # only use available features

    X = df[feature_cols]
    y = df['consumption_kwh']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(
        f"Training XGBoost model with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    logging.info("Evaluating XGBoost model...")
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    logging.info(f"Feature Level {feature_level} - MAE: {mae:.4f}")
    logging.info(f"Feature Level {feature_level} - RMSE: {rmse:.4f}")
    logging.info(f"Feature Level {feature_level} - R2 Score: {r2:.4f}")

    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": preds
    })

    return model, mae, rmse, r2, predictions_df


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

    logging.info(f"Loading electricity data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=['timestamp'])

    # Feature engineering
    df = apply_feature_level(df, level=FEATURE_LEVEL)

    # Encode building_id
    df, building_id_encoder = encode_building_id(df)

    # Train model
    model, mae, rmse, r2, predictions_df = train_xgboost(df, FEATURE_LEVEL)

    save_training_outputs(model, building_id_encoder, predictions_df, FEATURE_LEVEL, mae, rmse, r2)


if __name__ == "__main__":
    main()
