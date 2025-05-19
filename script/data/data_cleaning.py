import logging
import pandas as pd
from pathlib import Path

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_csv(path, parse_dates=None):
    """Loads a CSV into a pandas DataFrame."""
    try:
        df = pd.read_csv(path, parse_dates=parse_dates)
        logging.info(f"Loaded dataset from {path} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

def missing_value_report(df, name):
    """Logs missing value percentage for each column."""
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        logging.info(f"No missing values detected in {name}.")
    else:
        logging.info(f"Missing Value Report for {name}:")
        logging.info(f"\n{missing}")

def drop_high_missing_meters(df, threshold=0.3):
    """Drops columns (meters) with missing value fraction above threshold."""
    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()

    logging.info(f"Dropping {len(cols_to_drop)} meters with >{threshold*100:.0f}% missing values.")
    return df.drop(columns=cols_to_drop)

def clean_metadata(df):
    """Selects relevant metadata fields and handles missingness."""
    selected_cols = ['building_id', 'site_id', 'primaryspaceusage', 'sqm', 'yearbuilt', 'lat', 'lng']
    df = df[selected_cols]

    logging.info("Cleaning metadata: dropping rows with missing lat/lng and imputing yearbuilt.")
    df = df.dropna(subset=['lat', 'lng'])
    median_year = df['yearbuilt'].median()
    df['yearbuilt'] = df['yearbuilt'].fillna(median_year)

    return df

def clean_weather(df):
    """Sorts and interpolates missing numeric weather values."""
    logging.info("Cleaning weather data: interpolating numeric columns.")
    df = df.sort_values(['site_id', 'timestamp'])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

    return df

def save_dataframe(df, path):
    """Saves a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Saved cleaned dataset to {path}.")

def main():
    setup_logger()

    # Define paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data/raw"
    interim_dir = base_dir / "data/interim"
    processed_dir = base_dir / "data/processed"

    electricity_path = raw_dir / "electricity.csv"
    metadata_path = raw_dir / "metadata.csv"
    weather_path = raw_dir / "weather.csv"

    # Load data
    electricity_df = load_csv(electricity_path, parse_dates=['timestamp'])
    metadata_df = load_csv(metadata_path)
    weather_df = load_csv(weather_path, parse_dates=['timestamp'])

    if any(df is None for df in [electricity_df, metadata_df, weather_df]):
        logging.error("Failed to load one or more datasets. Exiting.")
        return

    # Report missing values
    missing_value_report(electricity_df, "Electricity")
    missing_value_report(metadata_df, "Metadata")
    missing_value_report(weather_df, "Weather")

    # Cleaning steps
    electricity_clean = drop_high_missing_meters(electricity_df)
    metadata_clean = clean_metadata(metadata_df)
    weather_clean = clean_weather(weather_df)

    # Save cleaned data
    save_dataframe(electricity_clean, interim_dir / "electricity_data_clean.csv")
    save_dataframe(metadata_clean, processed_dir / "metadata_clean.csv")
    save_dataframe(weather_clean, processed_dir / "weather_clean.csv")

    # Summary
    logging.info("\nCleaned Data Shapes:")
    logging.info(f"Electricity: {electricity_clean.shape}")
    logging.info(f"Metadata: {metadata_clean.shape}")
    logging.info(f"Weather: {weather_clean.shape}")

if __name__ == "__main__":
    main()
