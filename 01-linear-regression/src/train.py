"""
Complete training pipeline with experiments.

This script:
1. Loads California Housing data
2. Trains multiple model configurations
3. Evaluates each model
4. Performs cross-validation
5. Generates visualizations
6. Creates comparison report
"""

from data import load_california_housing, split_and_scale, split_no_scaling, explore_dataset
from model import create_model
from evaluate import RegressionMetrics, cross_validate_model, print_cv_report, print_metrics_report
from visualize import (
    plot_predictions_vs_actual, plot_residuals, plot_feature_correlation,
    plot_error_distribution, plot_model_comparison
)
import pandas as pd
import numpy as np


def experiment_1_baseline():
    """
    Experiment 1: Baseline Linear Regression
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: BASELINE LINEAR REGRESSION")
    print("="*70)
    
    # Load data
    X, y = load_california_housing()
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Explore
    stats = explore_dataset(X, y)
    print(f"\nDataset Statistics:")
    print(f"  Target mean: {stats['y_mean']:.4f}")
    print(f"  Target std: {stats['y_std']:.4f}")
    print(f"  Target range: [{stats['y_min']:.4f}, {stats['y_max']:.4f}]")
    
    # Split
    X_train, X_test, y_train, y_test = split_no_scaling(X, y)
    
    # Train
    model = create_model('baseline')
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    metrics_train = RegressionMetrics.compute_all_metrics(y_train, y_pred_train)
    metrics_test = RegressionMetrics.compute_all_metrics(y_test, y_pred_test)
    
    print("\nTraining Metrics:")
    RegressionMetrics.print_metrics_report(y_train, y_pred_train)
    
    print("Test Metrics:")
    RegressionMetrics.print_metrics_report(y_test, y_pred_test)
    
    # Cross-validation
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    print("Cross-Validation Results (5-fold):")
    print_cv_report(cv_results)
    
    # Visualize
    plot_predictions_vs_actual(y_test, y_pred_test, 
                              title="Baseline: Predictions vs Actual (Test Set)",
                              save_path="exp1_predictions.png")
    plot_residuals(y_test, y_pred_test,
                  title="Baseline: Residual Analysis (Test Set)",
                  save_path="exp1_residuals.png")
    
    return {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "cv_results": cv_results,
    }


def experiment_2_scaling():
    """
    Experiment 2: Linear Regression with Feature Scaling
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: LINEAR REGRESSION WITH SCALING")
    print("="*70)
    
    # Load and split
    X, y = load_california_housing()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    # Train
    model = create_model('scaled')
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    metrics_test = RegressionMetrics.compute_all_metrics(y_test, y_pred_test)
    
    print("\nTest Metrics:")
    RegressionMetrics.print_metrics_report(y_test, y_pred_test)
    
    print(f"\nRMSE: {metrics_test['rmse']:.6f}")
    print(f"R² Score: {metrics_test['r2_score']:.6f}")
    
    return {
        "model": model,
        "y_pred_test": y_pred_test,
        "metrics_test": metrics_test,
    }


def experiment_3_log_transform():
    """
    Experiment 3: Linear Regression with Log Transformation
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: LINEAR REGRESSION WITH LOG TRANSFORMATION")
    print("="*70)
    
    # Load and split
    X, y = load_california_housing()
    X_train, X_test, y_train, y_test = split_no_scaling(X, y)
    
    # Train
    model = create_model('log_transform')
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    metrics_test = RegressionMetrics.compute_all_metrics(y_test, y_pred_test)
    
    print("\nTest Metrics:")
    RegressionMetrics.print_metrics_report(y_test, y_pred_test)
    
    print(f"\nRMSE: {metrics_test['rmse']:.6f}")
    print(f"R² Score: {metrics_test['r2_score']:.6f}")
    
    return {
        "model": model,
        "y_pred_test": y_pred_test,
        "metrics_test": metrics_test,
    }


def experiment_4_feature_engineering():
    """
    Experiment 4: Linear Regression with Feature Engineering
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: LINEAR REGRESSION WITH FEATURE ENGINEERING")
    print("="*70)
    
    # Load and split
    X, y = load_california_housing()
    X_train, X_test, y_train, y_test = split_no_scaling(X, y)
    
    # Train
    model = create_model('feature_engineering')
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    metrics_test = RegressionMetrics.compute_all_metrics(y_test, y_pred_test)
    
    print("\nTest Metrics:")
    RegressionMetrics.print_metrics_report(y_test, y_pred_test)
    
    print(f"\nRMSE: {metrics_test['rmse']:.6f}")
    print(f"R² Score: {metrics_test['r2_score']:.6f}")
    
    return {
        "model": model,
        "y_pred_test": y_pred_test,
        "metrics_test": metrics_test,
    }


def main():
    """Run all experiments."""
    print("\n" + "="*70)
    print("LINEAR REGRESSION EXPERIMENTS - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Load data once for correlation
    X, y = load_california_housing()
    
    # Run experiments
    exp1 = experiment_1_baseline()
    exp2 = experiment_2_scaling()
    exp3 = experiment_3_log_transform()
    exp4 = experiment_4_feature_engineering()
    
    # Feature correlation
    plot_feature_correlation(X, y, save_path="feature_correlation.png")
    
    # Model comparison
    results_dict = {
        "Baseline": exp1["metrics_test"],
        "Scaled": exp2["metrics_test"],
        "Log Transform": exp3["metrics_test"],
        "Feature Eng.": exp4["metrics_test"],
    }
    
    plot_model_comparison(results_dict, save_path="model_comparison.png")
    
    # Summary table
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    summary_data = {
        "Model": ["Baseline", "Scaled", "Log Transform", "Feature Eng."],
        "R² Score": [
            results_dict["Baseline"]["r2_score"],
            results_dict["Scaled"]["r2_score"],
            results_dict["Log Transform"]["r2_score"],
            results_dict["Feature Eng."]["r2_score"],
        ],
        "MAE": [
            results_dict["Baseline"]["mae"],
            results_dict["Scaled"]["mae"],
            results_dict["Log Transform"]["mae"],
            results_dict["Feature Eng."]["mae"],
        ],
        "RMSE": [
            results_dict["Baseline"]["rmse"],
            results_dict["Scaled"]["rmse"],
            results_dict["Log Transform"]["rmse"],
            results_dict["Feature Eng."]["rmse"],
        ],
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n")
    print(summary_df.to_string(index=False))
    print("\n")
    
    print("="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  - exp1_predictions.png")
    print("  - exp1_residuals.png")
    print("  - feature_correlation.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()
