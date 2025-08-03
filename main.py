import argparse
import os
import pandas as pd
import joblib
from features.credit_feature_extraction import CreditFeatureExtractor
from src.data_preprocessing import load_data, create_preprocessor, split_data, get_feature_lists
from src.eda import eda_plots
from src.model_pipeline import train_and_select_model
import warnings
warnings.filterwarnings('ignore')  

# ----------------------------
# Feature Creation
# ----------------------------
def run_feature_creation(json_path, output_dir):
    print("\n✅ Running Feature Creation...")
    os.makedirs(output_dir, exist_ok=True)
    extractor = CreditFeatureExtractor(json_path=json_path)
    features_df = extractor.create_features()
    output_file = os.path.join(output_dir, "credit_report_var.csv")
    features_df.to_csv(output_file, index=False)
    print(f"✅ Features extracted and saved at: {output_file}")
    return output_file

# ----------------------------
# Model Training
# ----------------------------
def run_model_training(data_path, tune=False, tune_method="grid"):
    print("\n✅ Running Model Training...")
    df = load_data(data_path)

    # Perform EDA and save plots in reports/eda
    eda_plots(df)

    # ✅ Dynamically detect numeric and categorical features
    numeric_features, categorical_features = get_feature_lists(df)
    print("\n✅ Numeric Features:", numeric_features)
    print("✅ Categorical Features:", categorical_features)

    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train and select best model (includes tuning + saving metrics + plots)
    best_model = train_and_select_model(
        X_train, y_train, X_test, y_test,
        preprocessor,
        tune=tune,
        tune_method=tune_method
    )

    # Save the best model in models/ directory as backup
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    print("✅ Backup model saved at models/best_model.pkl")

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Credit Risk Feature & Model Pipeline")
    parser.add_argument('--mode', choices=['feature_creation', 'model_training', 'all'], required=True,
                        help="Pipeline mode: feature_creation | model_training | all")
    parser.add_argument('--json', type=str, default='data/raw/credit_report_sample.json',
                        help="Path to raw credit report JSON file")
    parser.add_argument('--data', type=str, default='data/raw/credit.csv',
                        help="Path to credit dataset for model training")
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help="Directory to save processed features")
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning for models')
    parser.add_argument('--tune-method', type=str, choices=['grid', 'optuna'], default='grid',
                        help='Tuning method: grid or optuna')

    args = parser.parse_args()

    if args.mode == 'feature_creation':
        run_feature_creation(args.json, args.processed_dir)

    elif args.mode == 'model_training':
        run_model_training(args.data, tune=args.tune, tune_method=args.tune_method)

    elif args.mode == 'all':
        # 1. Feature Creation
        run_feature_creation(args.json, args.processed_dir)
        # 2. Model Training
        run_model_training(args.data, tune=args.tune, tune_method=args.tune_method)

if __name__ == "__main__":
    main()