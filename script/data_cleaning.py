import logging
import pandas as pd
from pathlib import Path

def clean_electricity_data(df, save_path=None):
    """
    Cleans the electricity data.
    If save_path is provided, saves the cleaned data to disk.

    Args:
        df (pd.DataFrame): Interim long-format electricity data.
        save_path (str, optional): If provided, save cleaned DataFrame to this path.

    Returns:
        pd.DataFrame: Cleaned electricity data.
    """
    logging.info("Starting data cleaning...")

    # Check and log issues
    missing_summary = df.isnull().sum()
    logging.info(f"Missing values per column:\n{missing_summary}")

    negative_values = df[df['consumption_kwh'] < 0]
    logging.info(f"Negative consumption records: {len(negative_values)}")

    zero_values = df[df['consumption_kwh'] == 0]
    logging.info(f"Zero consumption records: {len(zero_values)}")

    missing_records = df[df[['timestamp', 'building_id', 'consumption_kwh']].isnull().any(axis=1)]
    logging.info(f"Rows with missing critical fields: {len(missing_records)}")

    df_clean = df.dropna(subset=['timestamp', 'building_id', 'consumption_kwh']).reset_index(drop=True)

    logging.info(f"Data cleaning completed. Final shape: {df_clean.shape}")

    if save_path:
        df_clean.to_csv(save_path, index=False)
        logging.info(f"Cleaned data saved to {save_path}")

    return df_clean

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_PATH = BASE_DIR / "data/interim/electricity_data_long.csv"
    OUTPUT_PATH = BASE_DIR / "data/processed/electricity_data_long_clean.csv"

    logging.info(f"Loading interim long-format electricity data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, parse_dates=['timestamp'])

    df_clean = clean_electricity_data(df, save_path=OUTPUT_PATH)
    logging.info("Finished cleaning and saving electricity data.")
