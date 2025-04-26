import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_basic_time_features(df):
    """
    Add basic time-based features: hour, dayofweek, month, is_weekend.
    """
    logging.info("Adding basic time features...")
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    return df

def add_season_feature(df):
    """
    Add season categorical feature (Winter, Spring, Summer, Fall) and numeric encoding.
    """
    logging.info("Adding season feature...")
    if 'timestamp' not in df.columns:
        raise ValueError("Timestamp column required for season feature.")

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['timestamp'].dt.month.apply(get_season)
    season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
    df['season_encoded'] = df['season'].map(season_mapping)
    return df

def apply_feature_level(df, level=1):
    logging.info(f"Applying feature engineering at level {level}")

    BASE_DIR = Path(__file__).resolve().parent.parent

    if level >= 1:
        df = add_basic_time_features(df)
    if level >= 2:
        df = add_season_feature(df)
    if level >= 3:
        logging.info("Loading and merging metadata...")
        METADATA_PATH = BASE_DIR / "data/raw/metadata.csv"
        metadata = pd.read_csv(METADATA_PATH)

        for col in ['building_id', 'site_id', 'sqm', 'yearbuilt', 'primaryspaceusage', 'lat', 'lng']:
            if col not in metadata.columns:
                metadata[col] = -1

        if 'primaryspaceusage' in metadata.columns:
            metadata['primaryspaceusage'] = metadata['primaryspaceusage'].fillna('Unknown')
            primary_use_encoder = LabelEncoder()
            metadata['primaryspaceusage_encoded'] = primary_use_encoder.fit_transform(metadata['primaryspaceusage'])
        else:
            metadata['primaryspaceusage_encoded'] = -1

        metadata = metadata.rename(columns={'lat': 'latitude', 'lng': 'longitude', 'yearbuilt': 'year_built', 'primaryspaceusage_encoded': 'primary_space_usage_encoded'})
        metadata = metadata[['building_id', 'site_id', 'sqm', 'year_built', 'primary_space_usage_encoded', 'latitude', 'longitude']]

        for col in ['sqm', 'year_built', 'latitude', 'longitude']:
            metadata[col] = metadata[col].fillna(-1)

        df = df.merge(metadata, on='building_id', how='left')
        df['primary_space_usage_encoded'] = df['primary_space_usage_encoded'].fillna(-1).astype(int)
        df['year_built'] = df['year_built'].fillna(-1).astype(int)

    if level >= 4:
        logging.info("Loading and merging weather features...")
        WEATHER_PATH = BASE_DIR / "data/raw/weather.csv"
        weather = pd.read_csv(WEATHER_PATH, parse_dates=['timestamp'])

        # 🛡 Fix: Make df timestamps naive if needed
        if pd.api.types.is_datetime64tz_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'].dt.tz_convert(None)

        weather = weather.drop_duplicates(subset=['timestamp', 'site_id'])
        df = df.merge(weather, on=['timestamp', 'site_id'], how='left')

        for col in weather.columns:
            if col not in ['timestamp', 'site_id']:
                df[col] = df[col].fillna(df[col].mean())

    return df

def get_feature_columns_by_level(level):
    base_features = ['hour', 'dayofweek', 'month', 'is_weekend']
    if level >= 2:
        base_features += ['season_encoded']
    if level >= 3:
        base_features += ['sqm', 'year_built', 'primary_space_usage_encoded', 'latitude', 'longitude']
    if level >= 4:
        weather_features = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth', 'wind_speed']
        base_features += weather_features
    return base_features

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    sample_path = BASE_DIR / "data/processed/electricity_data_long_clean.csv"
    df = pd.read_csv(sample_path, parse_dates=['timestamp'])
    df = apply_feature_level(df, level=4)
    logging.info(df.head())
