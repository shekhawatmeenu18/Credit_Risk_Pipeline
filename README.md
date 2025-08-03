# Credit Risk Prediction Pipeline

This project implements an end-to-end pipeline for:
1. Feature extraction from credit report JSON
2. Credit risk modeling using Logistic Regression, Random Forest, and XGBoost
3. Hyperparameter tuning (GridSearchCV or Optuna)
4. Feature importance analysis (plots + CSV)
5. EDA with visualization
6. Saving model performance reports


## Project Structure
```plaintext
fairmoney/
│
├── data/
│ ├── raw/ # Raw input data
│ ├── processed/ # Processed feature data
│
├── features/
│ ├── credit_feature_extraction.py # Class to create features from JSON file
│ ├── credit_report_feature_creation.py # Extracted features functions
│
├── reports/
│ ├── eda/ # EDA visualizations
│ ├── feature_importance/ # Feature importance plots & CSVs
│ ├── model_performance/ # Model performance reports
│
├── models/ # Saved ML models
│
├── src/
│ ├── data_preprocessing.py
│ ├── eda.py
│ ├── model_pipeline.py
│
├── main.py # Main script to run pipeline
├── requirements.txt # Python dependencies
└── README.md
```


## Installation
1. Create virtual environment
python3 -m venv ml_env
source ml_env/bin/activate

2. Install dependencies
pip install -r requirements.txt

## Pipeline Run Instructions
1. Feature Creation Only
python main.py --mode feature_creation --json data/raw/credit_report_sample.json --processed_dir data/processed

2. Model Training only
python main.py --mode model_training --data data/raw/credit.csv

3. Enable Hyperparamter training
python main.py --mode model_training --data data/raw/credit.csv --tune --tune-method grid
python main.py --mode model_training --data data/raw/credit.csv --tune --tune-method optuna

4. Run Full Pipeline
python main.py --mode all --json data/raw/credit_report_sample.json --data data/raw/credit.csv --tune


## Output Files
1. Processed Features: data/processed/credit_report_var.csv
2. EDA Plots: reports/eda/
3. Feature Importance Plots & CSV: reports/feature_importance/
4. Model Performance Reports: reports/model_performance/
5. Saved Model: models/best_model.pkl


## Further Possible Enhancements
1. Add SHAP for advanced feature importance
2. Add automated model monitoring & drift detection