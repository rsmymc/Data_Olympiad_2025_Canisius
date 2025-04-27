import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def extract_datetime_features(df, timestamp_col='timestamp'):
    """
    Extracts datetime-based features from the timestamp column.

    Args:
        df (DataFrame): DataFrame containing a timestamp column.
        timestamp_col (str): Name of the timestamp column.

    Returns:
        DataFrame with new datetime features added.
    """
    logging.info("Extracting datetime features...")
    df['hour'] = df[timestamp_col].dt.hour
    df['dayofweek'] = df[timestamp_col].dt.dayofweek
    df['month'] = df[timestamp_col].dt.month
    df['dayofyear'] = df[timestamp_col].dt.dayofyear
    return df


def reshape_electricity_data(df):
    """
    Reshapes electricity data from wide to long format.

    Args:
        df (DataFrame): Wide format electricity data.

    Returns:
        DataFrame in long format with 'timestamp', 'building_id', and 'meter_reading'.
    """
    logging.info("Reshaping electricity data from wide to long format...")
    id_vars = ['timestamp']
    value_vars = [col for col in df.columns if col not in id_vars]

    long_df = df.melt(id_vars=id_vars, value_vars=value_vars,
                      var_name='building_id', value_name='meter_reading')
    return long_df


def merge_datasets(electricity_df, metadata_df, weather_df):
    """
    Merges electricity readings with building metadata and weather data.

    Args:
        electricity_df (DataFrame): Long format electricity readings.
        metadata_df (DataFrame): Building metadata.
        weather_df (DataFrame): Weather data.

    Returns:
        Merged DataFrame.
    """
    logging.info("Merging datasets (electricity + metadata + weather)...")
    merged_df = electricity_df.merge(metadata_df, how='left', on='building_id')

    if merged_df['timestamp'].dt.tz is not None:
        merged_df['timestamp'] = merged_df['timestamp'].dt.tz_localize(None)

    merged_df = merged_df.merge(weather_df, how='left', on=['site_id', 'timestamp'])
    return merged_df


def encode_categoricals(df, cols_to_encode):
    logging.info("Encoding categorical variables...")
    for col in cols_to_encode:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
    return df


def create_lag_features(df, groupby_cols, target_col, lags):
    logging.info("Creating lag features...")
    df = df.sort_values(groupby_cols + ['timestamp'])
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby(groupby_cols)[target_col].shift(lag)
    return df


def create_rolling_features(df, groupby_cols, target_col, windows):
    logging.info("Creating rolling average features...")
    df = df.sort_values(groupby_cols + ['timestamp'])
    for window in windows:
        df[f"{target_col}_roll{window}"] = df.groupby(groupby_cols)[target_col].rolling(window=window,
                                                                                        min_periods=1).mean().reset_index(
            level=groupby_cols, drop=True)
    return df


def apply_log_transform(df, target_col='meter_reading'):
    logging.info(f"Applying log1p transformation to {target_col}...")
    df[f"log_{target_col}"] = np.log1p(df[target_col])
    return df


def aggregate_to_daily(df):
    logging.info("Aggregating to daily level...")
    df['date'] = df['timestamp'].dt.date
    daily_df = df.groupby(['building_id', 'date']).agg({
        'meter_reading': 'sum',
        'log_meter_reading': 'mean',
        'site_id': 'first',
        'primaryspaceusage': 'first',
        'sqm': 'first',
        'yearbuilt': 'first',
        'lat': 'first',
        'lng': 'first',
        'airTemperature': 'mean',
        'dewTemperature': 'mean',
        'seaLvlPressure': 'mean',
        'windSpeed': 'mean',
        'meter_reading_lag1': 'mean',
        'meter_reading_lag24': 'mean',
        'meter_reading_roll6': 'mean',
        'meter_reading_roll12': 'mean',
        'meter_reading_roll24': 'mean'
    }).reset_index()
    return daily_df


def summarize_merged_data(df):
    """
    Logs basic summary of the merged dataset.

    Args:
        df (DataFrame): Merged DataFrame.
    """
    logging.info("\nFull Merged Data Overview:")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"\n{df.head(3)}")


def plot_meter_reading_distribution(df, target_col='meter_reading', save_path=None):
    logging.info(f"Plotting distribution of {target_col}...")
    plt.figure(figsize=(8, 5))
    df[target_col].hist(bins=100, edgecolor='black')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.title(f"Distribution of {target_col}")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Plot saved to {save_path}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    interim_dir = base_dir / "data/interim"
    processed_dir = base_dir / "data/processed"
    plots_dir = base_dir / "plots"

    electricity_path = interim_dir / "electricity_data_clean.csv"
    metadata_path = processed_dir / "metadata_clean.csv"
    weather_path = processed_dir / "weather_clean.csv"

    # Load processed cleaned data
    electricity_df = pd.read_csv(electricity_path, parse_dates=['timestamp'])
    metadata_df = pd.read_csv(metadata_path)
    weather_df = pd.read_csv(weather_path, parse_dates=['timestamp'])

    # Apply feature engineering steps
    electricity_df = extract_datetime_features(electricity_df)
    electricity_long = reshape_electricity_data(electricity_df)
    full_df = merge_datasets(electricity_long, metadata_df, weather_df)

    # Encode categorical variables
    full_df = encode_categoricals(full_df, cols_to_encode=['primaryspaceusage', 'site_id'])

    # Create lag features (1 and 24 hours)
    full_df = create_lag_features(full_df, groupby_cols=['building_id'], target_col='meter_reading', lags=[1, 24])

    # Create rolling average features (6, 12, and 24 hours)
    full_df = create_rolling_features(full_df, groupby_cols=['building_id'], target_col='meter_reading',
                                      windows=[6, 12, 24])

    # Plot and save meter reading distribution
    plot_meter_reading_distribution(full_df, target_col='meter_reading',
                                    save_path=plots_dir / "meter_reading_distribution.png")

    # Apply log transformation
    full_df = apply_log_transform(full_df, target_col='meter_reading')

    # Plot and save meter reading distribution after transform
    plot_meter_reading_distribution(full_df, target_col='log_meter_reading',
                                    save_path=plots_dir / "log_meter_reading_distribution.png")

    # Aggregate to daily level
    full_df = aggregate_to_daily(full_df)

    # Drop NaN values after log-transform
    full_df = full_df.dropna(subset=['log_meter_reading'])

    # Summarize merged dataset
    summarize_merged_data(full_df)

    # Save full merged dataset
    full_merged_path = processed_dir / "electricity_clean_long.csv"
    full_df.to_csv(full_merged_path, index=False)
    logging.info(f"Saved full merged dataset to {full_merged_path}")


if __name__ == "__main__":
    main()
