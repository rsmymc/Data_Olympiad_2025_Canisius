# runner.py

import subprocess
import sys
import shutil
from pathlib import Path

def run_data_cleaning():
    print("\n🧹 Running Data Cleaning...")
    subprocess.run([sys.executable, "data_cleaning.py"])

def run_feature_engineering():
    print("\n🛠️ Running Feature Engineering...")
    subprocess.run([sys.executable, "feature_engineering.py"])

def run_xgboost_training():
    print("\n🚀 Running XGBoost Training...")
    subprocess.run([sys.executable, "xgboost_modelling.py"])

def run_neural_network_training():
    print("\n🧠 Running Neural Network Training...")
    subprocess.run([sys.executable, "neural_network_modelling.py"])

def run_plotting():
    print("\n🎨 Running Plots Generation...")
    subprocess.run([sys.executable, "analyzing.py"])

def clean_outputs():
    print("\n🧹 Cleaning models/, reports/, plots/ ...")
    paths_to_clean = ["models", "reports", "plots/xgboost", "plots/neural_network"]

    for folder_name in paths_to_clean:
        path = Path(folder_name)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            print(f"✅ Deleted {folder_name}/")
        else:
            print(f"⚠️ {folder_name}/ does not exist. Skipping.")

def main():
    print("""
    ============================
          Runner Menu
    ============================
    1. Data Cleaning
    2. Feature Engineering
    3. Train XGBoost
    4. Train Neural Network
    5. Generate Plots
    6. Full Pipeline (Clean → Feature Eng → Train → Plot)
    7. Clean Outputs (Reset models/reports/plots)
    8. Exit
    """)
    choice = input("Select an option (1-8): ")

    if choice == '1':
        run_data_cleaning()
    elif choice == '2':
        run_feature_engineering()
    elif choice == '3':
        run_xgboost_training()
    elif choice == '4':
        run_neural_network_training()
    elif choice == '5':
        run_plotting()
    elif choice == '6':
        run_data_cleaning()
        run_feature_engineering()
        run_xgboost_training()
        run_neural_network_training()
        run_plotting()
    elif choice == '7':
        clean_outputs()
    elif choice == '8':
        print("Exiting...")
    else:
        print("Invalid choice! Please select a valid option.")

if __name__ == "__main__":
    main()
