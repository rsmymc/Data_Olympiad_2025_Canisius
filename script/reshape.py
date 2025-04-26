import pandas as pd
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reshape_wide_to_long(input_path, save_path=None):
    """
    Reshape wide electricity data to long format.

    Args:
        input_path (str): Path to wide format CSV.
        save_path (str, optional): If provided, save long format CSV to this path.

    Returns:
        pd.DataFrame: Long format DataFrame.
    """
    logging.info(f"Loading wide format data from: {input_path}")
    df_wide = pd.read_csv(input_path)

    # 🔥 Add this for quick testing
    #df_wide = df_wide.head(1000)

    if 'timestamp' not in df_wide.columns:
        raise ValueError("Wide data must have a 'timestamp' column.")

    # Convert timestamp to datetime
    df_wide['timestamp'] = pd.to_datetime(df_wide['timestamp'])

    # Melt to long format
    df_long = df_wide.melt(
        id_vars=['timestamp'],
        var_name='building_id',
        value_name='consumption_kwh'
    )

    # Sort
    df_long = df_long.sort_values(by=['building_id', 'timestamp']).reset_index(drop=True)

    logging.info(f"Reshaped data: {df_long.shape[0]} rows, {df_long.shape[1]} columns.")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_long.to_csv(save_path, index=False)
        logging.info(f"Long format data saved to: {save_path}")

    return df_long

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_PATH = BASE_DIR / "data/raw/electricity_data_wide.csv"
    OUTPUT_PATH = BASE_DIR / "data/interim/electricity_data_long.csv"

    df_long = reshape_wide_to_long(INPUT_PATH, save_path=OUTPUT_PATH)
