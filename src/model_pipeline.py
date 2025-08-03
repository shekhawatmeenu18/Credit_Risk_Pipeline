import os
import json
import optuna
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


def plot_feature_importance(model, feature_names, model_name, output_dir="reports/feature_importance"):
    os.makedirs(output_dir, exist_ok=True)

    importance_data = None

    # Tree-based models: feature_importances_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        # Save Plot
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        plt.title(f"{model_name} Feature Importance")
        plt.bar(range(len(importances)), importances[indices], color="skyblue")
        plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{model_name}_importance.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to: {plot_path}")

    # Logistic Regression: coefficients
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
        importance_data = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)

        # Save Plot
        indices = np.argsort(np.abs(importances))[::-1]
        plt.figure(figsize=(12, 6))
        plt.title(f"{model_name} Feature Coefficients (Absolute Values)")
        plt.bar(range(len(importances)), np.abs(importances)[indices], color="salmon")
        plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{model_name}_coefficients.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature coefficients plot saved to: {plot_path}")

    # Save CSV
    if importance_data:
        df_importance = pd.DataFrame(importance_data, columns=["Feature", "Importance"])
        csv_path = os.path.join(output_dir, f"{model_name}_importance.csv")
        df_importance.to_csv(csv_path, index=False)
        print(f"Feature importance CSV saved to: {csv_path}")


def save_model_and_metrics(pipeline, X_test, y_test, model_name):
    """Save model and performance metrics in reports/model_performance."""
    os.makedirs("reports/model_performance", exist_ok=True)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True)
    }

    # Save metrics to JSON
    metrics_path = f"reports/model_performance/{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved at {metrics_path}")

    # Save model
    model_path = f"reports/model_performance/{model_name}_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved at {model_path}")


def train_and_select_model(X_train, y_train, X_test, y_test, preprocessor, tune=False, tune_method="grid"):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    results = {}
    pipelines = {}

    print("\n🔍 Training base models...")
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        results[name] = auc
        pipelines[name] = pipeline
        print(f"{name}: AUC = {auc:.4f}")

    # Select best model
    best_model_name = max(results, key=results.get)
    best_pipeline = pipelines[best_model_name]
    print(f"\n✅ Best model: {best_model_name} (AUC: {results[best_model_name]:.4f})")

    # Hyperparameter tuning (optional)
    if tune:
        print(f"\n🔍 Performing {tune_method} hyperparameter tuning on {best_model_name}...")
        if tune_method == "grid":
            best_pipeline = tune_model_grid(best_pipeline, best_model_name, X_train, y_train)
        elif tune_method == "optuna":
            best_pipeline = tune_model_optuna(best_pipeline, best_model_name, X_train, y_train)
        else:
            print("⚠ Invalid tune_method. Skipping tuning.")

    # Extract feature names from preprocessor
    feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Plot feature importance
    model = best_pipeline.named_steps['classifier']
    plot_feature_importance(model, feature_names, best_model_name)

    # ✅ Save model and metrics
    save_model_and_metrics(best_pipeline, X_test, y_test, best_model_name)

    return best_pipeline


def tune_model_grid(pipeline, model_name, X_train, y_train):
    param_grid = {}

    if model_name == "LogisticRegression":
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l2']
        }
    elif model_name == "RandomForest":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None]
        }
    elif model_name == "XGBoost":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7]
        }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params (GridSearch): {grid.best_params_}")
    return grid.best_estimator_


def tune_model_optuna(pipeline, model_name, X_train, y_train):
    def objective(trial):
        params = {}
        if model_name == "LogisticRegression":
            params['classifier__C'] = trial.suggest_float('classifier__C', 0.01, 10, log=True)
        elif model_name == "RandomForest":
            params['classifier__n_estimators'] = trial.suggest_int('classifier__n_estimators', 100, 300)
            params['classifier__max_depth'] = trial.suggest_int('classifier__max_depth', 3, 15)
        elif model_name == "XGBoost":
            params['classifier__n_estimators'] = trial.suggest_int('classifier__n_estimators', 100, 300)
            params['classifier__max_depth'] = trial.suggest_int('classifier__max_depth', 3, 10)

        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict_proba(X_train)[:, 1]
        return roc_auc_score(y_train, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print(f"Best params (Optuna): {study.best_params}")
    pipeline.set_params(**study.best_params)
    pipeline.fit(X_train, y_train)
    return pipeline