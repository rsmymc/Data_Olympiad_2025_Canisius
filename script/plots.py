import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data/processed/electricity_data_long_clean.csv"
EXPERIMENT_LOG_PATH = BASE_DIR / "results/xgboost_training_log.csv"
PREDICTIONS_PATH = BASE_DIR / "results"


def list_available_levels(model_type="xgboost"):
    model_files = list(MODELS_DIR.glob(f"{model_type}_model_level*.pkl"))
    levels = sorted(int(f.stem.split("level")[-1]) for f in model_files)
    return levels


def plot_feature_importance(level=None, model_type="xgboost"):
    levels = [level] if level is not None else list_available_levels(model_type)
    for lvl in levels:
        model_path = MODELS_DIR / f"{model_type}_model_level{lvl}.pkl"
        model = joblib.load(model_path)

        feature_names = model.get_booster().feature_names
        importance = model.feature_importances_

        # Sort by importance descending
        sorted_indices = importance.argsort()
        feature_names = [feature_names[i] for i in sorted_indices]
        importance = importance[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importance)
        plt.xlabel("Importance")
        plt.title(f"Feature Importance - Level {lvl}")
        plt.grid(True)
        plt.tight_layout()

        save_path = PLOTS_DIR / model_type / f"level{lvl}" / "feature_importance.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Feature importance plot saved to {save_path}")


def plot_experiment_results(experiment_log_path=EXPERIMENT_LOG_PATH, model_type="xgboost"):
    df = pd.read_csv(experiment_log_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['feature_level'], df['rmse'], marker='o', label='RMSE')
    plt.plot(df['feature_level'], df['mae'], marker='s', label='MAE')
    plt.xlabel("Feature Engineering Level")
    plt.ylabel("Error Metric (RMSE / MAE)")
    plt.title("Model Performance Across Feature Engineering Levels")
    plt.xticks(ticks=df['feature_level'], labels=[f"Level {int(lvl)}" for lvl in df['feature_level']])
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    save_path = PLOTS_DIR / model_type / "experiment_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Experiment results plot saved to {save_path}")


def plot_actual_vs_predicted(predictions_path=PREDICTIONS_PATH, level=None, model_type="xgboost"):
    levels = [level] if level is not None else list_available_levels(model_type)

    for lvl in levels:
        pred_path = predictions_path / f"{model_type}_predictions_level{lvl}.csv"
        df = pd.read_csv(pred_path)

        plt.figure(figsize=(8, 8))
        plt.scatter(df['y_true'], df['y_pred'], alpha=0.3)
        plt.xlabel("Actual Consumption (kWh)")
        plt.ylabel("Predicted Consumption (kWh)")
        plt.title(f"Actual vs Predicted - Level {lvl}")
        plt.plot([df['y_true'].min(), df['y_true'].max()],
                 [df['y_true'].min(), df['y_true'].max()],
                 'r--')
        plt.grid(True)
        plt.tight_layout()

        save_path = PLOTS_DIR / model_type / f"level{lvl}" / "actual_vs_predicted.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Actual vs Predicted plot saved to {save_path}")


def plot_error_distribution(predictions_path=PREDICTIONS_PATH, level=None, model_type="xgboost"):
    levels = [level] if level is not None else list_available_levels(model_type)

    for lvl in levels:
        pred_path = predictions_path / f"{model_type}_predictions_level{lvl}.csv"
        df = pd.read_csv(pred_path)

        errors = df['y_pred'] - df['y_true']

        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='k')
        plt.xlabel("Prediction Error (kWh)")
        plt.ylabel("Frequency")
        plt.title(f"Prediction Error Distribution - Level {lvl}")
        plt.grid(axis='y')
        plt.tight_layout()

        save_path = PLOTS_DIR / model_type / f"level{lvl}" / "error_distribution.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Error distribution plot saved to {save_path}")


if __name__ == "__main__":
    plot_feature_importance(level=None)
    plot_experiment_results()
    plot_actual_vs_predicted()
    plot_error_distribution()
