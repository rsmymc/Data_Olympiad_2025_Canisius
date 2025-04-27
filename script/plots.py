import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from xgboost import plot_importance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_PATH = BASE_DIR / "results"


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
        logging.info(f"Saved plot to {save_path}")


def plot_feature_importance(model, save_path=None):
    logging.info("Plotting feature importance...")
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=15, importance_type='gain')
    plt.title('XGBoost Feature Importance (Top 15)')
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Saved feature importance plot to {save_path}")
