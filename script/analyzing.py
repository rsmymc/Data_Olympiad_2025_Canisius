import logging
import matplotlib.pyplot as plt
import numpy as np
import shap
import joblib
import pandas as pd
import calendar
from pathlib import Path
from utils import split_train_test, load_predictions


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
    plt.close()


def plot_model_comparison(y_true, pred1, pred2, label1="XGBoost", label2="Neural Network", save_path=None):
    logging.info(f"Plotting Model Comparison between {label1} and {label2}...")
    y_true_inv = np.expm1(y_true)
    pred1_inv = np.expm1(pred1)
    pred2_inv = np.expm1(pred2)

    plt.figure(figsize=(12, 6))
    plt.plot(y_true_inv[:500], label='Actual', color='black', linewidth=2)
    plt.plot(pred1_inv[:500], label=label1, alpha=0.7)
    plt.plot(pred2_inv[:500], label=label2, alpha=0.7)
    plt.legend()
    plt.title('Model Predictions Comparison (First 500 Samples)')
    plt.xlabel('Samples')
    plt.ylabel('Energy Consumption (kWh)')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Saved model comparison plot to {save_path}")
    plt.close()


def plot_shap_summary(model, X_sample, model_name="Model", save_path=None):
    logging.info(f"Plotting SHAP Summary for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=5)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved SHAP Summary plot to {save_path}")
    plt.close()


def plot_residuals_sample(y_true_sample, y_pred_sample, model_name="Model (Sampled)", save_dir=None):
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
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "residuals_vs_predicted_sample.png")
        logging.info(f"Saved residuals vs predicted plot to {save_dir}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='blue', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Residual (kWh)')
    plt.ylabel('Frequency')
    plt.title(f'Residuals Distribution: {model_name}')
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "residuals_distribution_sample.png")
        logging.info(f"Saved residuals distribution plot to {save_dir}")
    plt.close()


def simulate_energy_savings(model, X_test, cost_per_kwh=0.12):
    """
    Simulates energy and cost savings scenarios based on model predictions.

    Args:
        model: Trained machine learning model (e.g., XGBoost, Neural Network).
        X_test (DataFrame): Test feature set.
        cost_per_kwh (float): Cost of one kilowatt-hour in USD (default is $0.12).

    Returns:
        savings_summary (DataFrame): Summary table of energy and cost savings.
    """
    logging.info("Simulating energy savings scenarios...")

    # Predict energy consumption (log scale), then inverse transform
    y_pred_log = model.predict(X_test)
    y_pred_kwh = np.expm1(y_pred_log)

    # Calculate total predicted energy
    total_predicted_energy = y_pred_kwh.sum()
    logging.info(f"Total Predicted Energy Consumption: {total_predicted_energy:,.2f} kWh")

    # Simulate 10% and 20% reductions
    saving_10_percent = total_predicted_energy * 0.10
    saving_20_percent = total_predicted_energy * 0.20

    new_total_10 = total_predicted_energy - saving_10_percent
    new_total_20 = total_predicted_energy - saving_20_percent

    # Calculate cost savings
    cost_saving_10_percent = saving_10_percent * cost_per_kwh
    cost_saving_20_percent = saving_20_percent * cost_per_kwh

    # Create a savings summary table
    savings_summary = pd.DataFrame({
        "Scenario": ["Baseline (No Reduction)", "10% Reduction", "20% Reduction"],
        "Total Predicted Energy (kWh)": [total_predicted_energy, new_total_10, new_total_20],
        "Energy Saved (kWh)": [0, saving_10_percent, saving_20_percent],
        "Estimated Cost Savings (USD)": [0, cost_saving_10_percent, cost_saving_20_percent]
    })

    logging.info("\n" + savings_summary.to_string(index=False))

    return savings_summary


def detect_underperforming_buildings(X_test, y_test, y_pred_log, top_n=10):
    """
    Detects underperforming buildings where actual energy use is much higher than predicted.

    Args:
        X_test (DataFrame): Test feature set (must contain 'building_id').
        y_test (Series): Actual log-transformed target values.
        y_pred_log (array): Predicted log-transformed target values.
        top_n (int): Number of top anomalies to return.

    Returns:
        top_underperformers (DataFrame): Top buildings with highest positive residuals.
    """
    logging.info("Detecting anomalous buildings...")

    # Inverse log1p to get actual kWh
    y_test_kwh = np.expm1(y_test)
    y_pred_kwh = np.expm1(y_pred_log)

    # Calculate residuals
    residuals = y_test_kwh - y_pred_kwh

    # Attach residuals to building info
    buildings = X_test.copy()
    buildings['actual_kwh'] = y_test_kwh
    buildings['predicted_kwh'] = y_pred_kwh
    buildings['residual'] = residuals

    # Aggregate residuals by building_id
    building_residuals = buildings.groupby('building_id').agg({
        'residual': 'mean',
        'actual_kwh': 'mean',
        'predicted_kwh': 'mean'
    }).reset_index()

    # Sort by highest positive residuals (underperformers)
    building_residuals = building_residuals.sort_values('residual', ascending=False)

    # Top N underperformers
    top_underperformers = building_residuals.head(top_n)

    logging.info(f"Top {top_n} Buildings Exhibiting Anomalous Energy Consumption:\n{top_underperformers.to_string(index=False)}")

    return top_underperformers


def plot_weather_feature_contribution(xgb_model, X_train, save_path=None):
    """
    Analyzes and plots weather feature contributions using SHAP values.

    Args:
        xgb_model: Trained XGBoost model.
        X_train (DataFrame): Training feature set.
        save_path (Path, optional): Path to save the plot.

    Returns:
        weather_contribution_percent (Series): Weather feature contribution percentages.
    """
    logging.info("Analyzing weather feature contributions with SHAP...")

    # Sample 800 points
    X_sample = X_train.sample(n=800, random_state=42)

    # Predict and Explain
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Focus on weather features
    weather_features = ['airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed']
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    shap_weather = shap_df[weather_features].abs().mean().sort_values(ascending=False)

    # Sum of all weather SHAP mean values
    total_weather_shap = shap_weather.sum()

    # Calculate each feature's % contribution
    weather_contribution_percent = (shap_weather / total_weather_shap) * 100
    weather_contribution_percent_df = weather_contribution_percent.sort_values(ascending=True)

    # Plot
    plt.figure(figsize=(8, 5))
    weather_contribution_percent_df.plot(kind='barh', color='skyblue')
    plt.xlabel('Contribution to Weather Sensitivity (%)')
    plt.title('Weather Feature Contribution to Energy Consumption')
    plt.xlim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Saved weather contribution plot to {save_path}")
    plt.close()

    logging.info("\nWeather Feature Contribution Percentages (%):")
    logging.info(f"\n{weather_contribution_percent.round(2).to_string()}")

    return weather_contribution_percent


def plot_seasonal_energy_trends(full_df_daily, save_path=None):
    """
    Plots seasonal energy consumption trends based on monthly aggregation.

    Args:
        full_df_daily (DataFrame): Full daily energy dataset (must contain 'timestamp' and 'meter_reading').
        save_path (Path, optional): Path to save the plot.

    Returns:
        monthly_energy (DataFrame): Aggregated monthly energy consumption.
    """
    logging.info("Plotting Seasonal Energy Consumption Trends...")

    # Ensure timestamp is datetime
    full_df_daily['date'] = pd.to_datetime(full_df_daily['date'])

    # Extract month
    full_df_daily['month'] = full_df_daily['date'].dt.month

    # Aggregate energy consumption by month
    monthly_energy = full_df_daily.groupby('month')['meter_reading'].sum().reset_index()

    # Map month number to full month name
    monthly_energy['month_name'] = monthly_energy['month'].apply(lambda x: calendar.month_name[x])

    # Sort by calendar order (just to be safe)
    monthly_energy = monthly_energy.sort_values('month')

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_energy['month_name'], monthly_energy['meter_reading'], marker='o', linestyle='-')
    plt.title('Seasonal Energy Consumption Trends')
    plt.xlabel('Month')
    plt.ylabel('Total Energy Consumption (kWh)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Saved seasonal energy trends plot to {save_path}")
    plt.close()

    return monthly_energy


def simulate_hotter_summer_impact(model, X_test, full_df_daily, summer_temp_increase=8, cost_per_kwh=0.12):
    """
    Simulates impact of a hotter summer (+X°C during June–August) on energy consumption and cost.

    Args:
        model: Trained model (e.g., XGBoost, Neural Network).
        X_test (DataFrame): Test feature set.
        full_df_daily (DataFrame): Full dataset to extract correct dates.
        summer_temp_increase (float): Degrees Celsius to simulate increase.
        cost_per_kwh (float): Cost of 1 kilowatt-hour in USD.

    Returns:
        summary_df (DataFrame): Summary table comparing baseline vs hotter summer scenario.
    """
    logging.info(f"Simulating a {summer_temp_increase}°C increase during Summer (June–August)...")

    # Copy and add month
    X_sim = X_test.copy()
    X_sim['month'] = pd.to_datetime(full_df_daily.loc[X_test.index, 'date']).dt.month

    # Increase airTemperature only in summer months
    summer_months = [6, 7, 8]
    summer_mask = X_sim['month'].isin(summer_months)
    X_sim.loc[summer_mask, 'airTemperature'] += summer_temp_increase

    # Drop helper column
    X_sim = X_sim.drop(columns='month')

    # Predict baseline and simulated
    baseline_kwh = np.expm1(model.predict(X_test)).sum()
    simulated_kwh = np.expm1(model.predict(X_sim)).sum()

    # Cost calculations
    baseline_cost = baseline_kwh * cost_per_kwh
    simulated_cost = simulated_kwh * cost_per_kwh

    # Differences
    delta_kwh = simulated_kwh - baseline_kwh
    delta_cost = simulated_cost - baseline_cost

    # Summary
    summary_df = pd.DataFrame({
        "Scenario": ["Baseline", f"Hotter Summer (+{summer_temp_increase}°C)"],
        "Total Energy (kWh)": [baseline_kwh, simulated_kwh],
        "Total Cost (USD)": [baseline_cost, simulated_cost],
        "Energy Increase (kWh)": [0, delta_kwh],
        "Cost Increase (USD)": [0, delta_cost],
        "Energy % Increase": [0, (delta_kwh / baseline_kwh) * 100],
        "Cost % Increase": [0, (delta_cost / baseline_cost) * 100]
    })

    logging.info("Cooling Load Stress Test Summary:\n" + summary_df.to_string(index=False))

    return summary_df



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    processed_dir = base_dir / "data/processed"
    plots_dir = base_dir / "plots"
    plots_dir_xgb = plots_dir / "xgboost"
    plots_dir_nn = plots_dir / "neural_network"
    plots_dir_cmp = plots_dir / "model_comparison"

    model_path = models_dir / "xgboost_model.pkl"
    if not model_path.exists():
        logging.error("Trained model not found!")
        exit()
    xgb_model = joblib.load(model_path)

    preds_df = load_predictions(reports_dir, "xgboost_predictions.csv")
    xgb_y_true = preds_df['y_true']
    xgb_y_pred = preds_df['y_pred']

    nn_preds_df = load_predictions(reports_dir, "nn_predictions.csv")
    nn_y_true = nn_preds_df['y_true']
    nn_y_preds = nn_preds_df['y_pred']

    daily_data_path = processed_dir / "electricity_clean_long.csv"
    full_df_daily = pd.read_csv(daily_data_path, parse_dates=['date'])

    X_train, X_test, y_train, y_test = split_train_test(full_df_daily, split_date='2017-01-01')
    #  Randomly sample from fully processed X_train
    X_sample = X_train.sample(n=800, random_state=42)
    y_sample = y_train.loc[X_sample.index]
    y_pred_sample = xgb_model.predict(X_sample)

    # plot_actual_vs_predicted(xgb_y_true, xgb_y_pred, model_name="XGBoost", save_path=plots_dir_xgb / "actual_vs_predicted.png")
    # plot_actual_vs_predicted(nn_y_true, nn_y_preds, model_name="Neural Network", save_path=plots_dir_nn / "actual_vs_predicted.png")
    # plot_model_comparison(xgb_y_true, xgb_y_pred, nn_y_preds, label1="XGBoost", label2="Neural Network", save_path=plots_dir_cmp / "xgb_vs_nn_comparison.png")
    # plot_shap_summary(xgb_model, X_sample, model_name="XGBoost", save_path=plots_dir_xgb / "shap_summary.png")
    # plot_residuals_sample(y_sample, y_pred_sample, model_name="XGBoost Sampled", save_dir=plots_dir_xgb)

    #savings_summary = simulate_energy_savings(xgb_model, X_test)
    #savings_summary.to_csv(reports_dir / "savings_summary.csv", index=False)

    # Set Predict log values
    # y_pred_log = xgb_model.predict(X_test)
    # # Inverse log transform for real-world kWh
    # y_pred_kwh = np.expm1(y_pred_log)
    # # Reattach building_id to X_test for analysis
    # X_test_with_id = X_test.copy()
    # X_test_with_id['building_id'] = full_df_daily.loc[X_test.index, 'building_id'].values
    # top_underperformers = detect_underperforming_buildings(X_test_with_id, y_test, y_pred_log)

    # weather_contribution = plot_weather_feature_contribution(
    #     xgb_model,
    #     X_train,
    #     save_path=plots_dir_xgb / "weather_contribution.png"
    # )

    monthly_energy = plot_seasonal_energy_trends(full_df_daily, save_path=plots_dir_xgb / "seasonal_energy_trends.png")

    summer_simulation_8c_result = simulate_hotter_summer_impact(
        model=xgb_model,
        X_test=X_test,
        full_df_daily=full_df_daily,
        summer_temp_increase=8,
        cost_per_kwh=0.12
    )
    summer_simulation_8c_result.to_csv(reports_dir / "Cooling Load Stress Test Summary.csv", index=False)
    logging.info(f"Saved Summer Impact Simulation")