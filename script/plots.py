import logging
import matplotlib.pyplot as plt
import numpy as np
import shap
import joblib
import pandas as pd
from pathlib import Path

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model", save_path=None):
    logging.info(f"Plotting Actual vs Predicted for {model_name}...")
    y_true_inv = np.expm1(y_true)
    y_pred_inv = np.expm1(y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_inv, y_pred_inv, alpha=0.3, s=5)
    plt.plot([y_true_inv.min(), y_true_inv.max()], [y_true_inv.min(), y_true_inv.max()], 'r--')
    plt.xlabel('Actual Energy Consumption (kWh)')
    plt.ylabel('Predicted Energy Consumption (kWh)')
    plt.title(f'Actual vs Predicted: {model_name}')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Saved  Actual vs Predicted plot to {save_path}")


def plot_residuals_sample(y_true_sample, y_pred_sample, model_name="Model (Sampled)", save_path=None):
    logging.info(f"Plotting Residuals for {model_name} on sample...")
    y_true_inv = np.expm1(y_true_sample)
    y_pred_inv = np.expm1(y_pred_sample)
    residuals = y_true_inv - y_pred_inv

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_inv, residuals, alpha=0.3, s=5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Energy Consumption (kWh)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residuals vs Predicted: {model_name}')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"residuals_vs_predicted_sample")
        logging.info(f"Saved residuals vs predicted plot to {save_path}")

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='blue', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residual (kWh)')
    plt.ylabel('Frequency')
    plt.title(f'Residuals Distribution: {model_name}')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / f"residuals_distribution_sample")
        logging.info(f"Saved residuals distribution plot to {save_path}")


def plot_shap_summary(model, X_test, save_path=None):
    logging.info("Plotting SHAP summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=5)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved SHAP Summary plot to {save_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    processed_dir = base_dir / "data/processed"
    plots_dir = base_dir / "plots"

    # Load model
    model_path = models_dir / "xgboost_model.pkl"
    if not model_path.exists():
        logging.error("Trained model not found!")
        exit()
    model = joblib.load(model_path)

    # Load predictions
    preds_path = reports_dir / "xgboost_predictions.csv"
    if not preds_path.exists():
        logging.error("Predictions file not found!")
        exit()
    preds_df = pd.read_csv(preds_path)

    y_true = np.log1p(preds_df['y_true'])
    y_pred = np.log1p(preds_df['y_pred'])

    # Load features for SHAP
    daily_data_path = processed_dir / "electricity_clean_long.csv"
    full_df_daily = pd.read_csv(daily_data_path, parse_dates=['date'])
    feature_cols = [
        'site_id', 'primaryspaceusage', 'sqm', 'yearbuilt', 'lat', 'lng',
        'airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed',
        'meter_reading_lag1', 'meter_reading_lag24', 'meter_reading_roll6',
        'meter_reading_roll12', 'meter_reading_roll24'
    ]
    split_date = '2017-01-01'
    X_test = full_df_daily[full_df_daily['date'] >= split_date][feature_cols]

    # Call plots
    #plot_actual_vs_predicted(y_true, y_pred, model_name="XGBoost", save_path=plots_dir / "xgboost" / "actual_vs_predicted.png")
    #plot_shap_summary(model, X_test, save_path=plots_dir / "xgboost" / "shap_summary.png")

    # Sample plotting
    sample_idx = np.random.choice(len(X_test), size=min(5000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    y_sample = y_true.iloc[sample_idx]
    y_pred_sample = model.predict(X_sample)

    plot_residuals_sample(y_sample, y_pred_sample, model_name="XGBoost_Sampled", save_path=plots_dir / "xgboost")