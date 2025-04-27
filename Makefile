# Makefile

# Define python
PYTHON=python

# Install all required packages
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

#Data Cleaning and Feature Engineering
data_cleaning:
	$(PYTHON) data_cleaning.py

feature_engineering:
	$(PYTHON) feature_engineering.py

# Model Training
train_xgb:
	$(PYTHON) xgboost_modelling.py

train_nn:
	$(PYTHON) neural_network_modelling.py

# Plotting
plots:
	$(PYTHON) analyzing.py

# Full pipeline
all: install data_cleaning feature_engineering train_xgb train_nn plots

# Clean all outputs
clean:
	rm -rf models reports plots
	echo "🧹 Cleaned models/, reports/, plots/ folders!"
