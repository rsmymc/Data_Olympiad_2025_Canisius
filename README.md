# Energy Consumption Forecasting and Analysis

This project models and analyzes building energy consumption using machine learning techniques, 
including XGBoost and Neural Networks. It also simulates energy savings and sensitivity to external factors like temperature increases.

---

## 📦 Project Structure

```bash
├── data/
│   ├── raw/            # Original input datasets
│   ├── interim/        # Cleaned and partially processed data
│   ├── processed/      # Fully processed data ready for modeling
├── models/             # Saved machine learning models
├── reports/            # Evaluation metrics, predictions, simulation results
├── plots/              # Visualization outputs
│   ├── xgboost/
│   ├── neural_network/
│   ├── model_comparison/
├── scripts/             # All Python scripts
│   ├── utils.py                # Utility functions (splitting, loading)
│   ├── Genome_P2.R             #
│   ├── data_cleaning.py        # Load raw data, clean missing values, process metadata 
│   ├── feature_engineering.py  # Create datetime features, lag/rolling features, merge datasets
│   ├── xgboost_modelling.py
│   ├── neural_network_modelling.py
│   ├── analysis.py             # For visualization and analysis
│   ├── runner.py               # Full pipeline runner script (interactive menu)
├── Makefile            # Quick automation commands
├── requirements.txt    # Required packages
└── README.md           # Project overview
```

---

## 🚀 Setup Instructions

1. **Install Requirements:**

```bash
pip install -r requirements.txt
```

2. **Run Full Pipeline with Menu:**

```bash
python runner.py
```

or use **Makefile**:

```bash
make all
```

---

## 🧠 Core Scripts

| Script                        | Purpose                                                                           |
|:------------------------------|:----------------------------------------------------------------------------------|
| `Genome_P2.R`                 | !!!Explanantation!!!                                                              |
| `data_cleaning.py`            | Load raw data, clean missing values, process metadata                             |
| `feature_engineering.py`      | Create datetime features, lag/rolling features, merge datasets                    |
| `xgboost_modelling.py`        | Train XGBoost model, evaluate, save predictions                                   |
| `neural_network_modelling.py` | Train Neural Network model, evaluate, save predictions                            |
| `analysis.py`                 | Visualizations, SHAP explainability, simulations (energy savings, hotter summers) |
| `runner.py`                   | Menu-based full pipeline runner                                                   |

---

## 📊 Key Features

- **Time-Based Splitting** (before/after 2017-01-01)
- **XGBoost and Neural Network Comparison**
- **SHAP Explainability** for feature importance
- **Residual Analysis** to detect anomalies
- **Energy Savings Simulation** (10%, 20% reduction scenarios)
- **Hotter Summer Impact Simulation** (+8°C stress testing)
- **Model Performance Evaluation** (RMSE, MAE, R²)

---

## 📈 Example Outputs

| Plot | Description |
|:---|:---|
| Actual vs Predicted Scatter | Model prediction accuracy |
| Residual Distribution | Error distribution analysis |
| SHAP Summary | Feature contribution explainability |
| Seasonal Energy Trend | Monthly energy patterns |
| Hotter Summer Simulation | Impact of +8°C on cooling load |

---

## 💬 Notes

- All outputs (plots, reports) are automatically saved into `plots/` and `reports/`.
- Flexible and modular code: easy to add new models or simulations later.

---

## ✨ Future Work

- Incorporate weather forecast data for future predictions
- !!!ADD!!!

---

## 👨‍💻 Contributors

- Julia Hise (hisej@canisius.edu)
- Edidiong Ibokete (iboketee@canisius.edu)
- Nuzaif Naveed (naveedn@canisius.edu)
- Rasim Yamac (yamacr@canisius.edu)

---
