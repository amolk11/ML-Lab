"""
Evaluation metrics and analysis for Linear Regression.

This module provides:
- Multiple evaluation metrics
- Residual analysis
- Error decomposition
- Cross-validation setup
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple


class RegressionMetrics:
    """Comprehensive metrics for regression models."""
    
    @staticmethod
    def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute basic regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2_score": r2_score(y_true, y_pred),
            "median_ae": median_absolute_error(y_true, y_pred),
        }
    
    @staticmethod
    def compute_residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute residual statistics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary with residual stats
        """
        residuals = y_true - y_pred
        
        return {
            "residual_mean": np.mean(residuals),
            "residual_std": np.std(residuals),
            "residual_min": np.min(residuals),
            "residual_max": np.max(residuals),
            "residual_q25": np.percentile(residuals, 25),
            "residual_q75": np.percentile(residuals, 75),
            "residual_iqr": np.percentile(residuals, 75) - np.percentile(residuals, 25),
        }
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute all available metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Combined metrics dictionary
        """
        metrics = {}
        metrics.update(RegressionMetrics.compute_basic_metrics(y_true, y_pred))
        metrics.update(RegressionMetrics.compute_residual_stats(y_true, y_pred))
        return metrics
    
    @staticmethod
    def print_metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Print formatted metrics report.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        """
        metrics = RegressionMetrics.compute_all_metrics(y_true, y_pred)
        
        print("\n" + "="*60)
        print("REGRESSION METRICS REPORT")
        print("="*60)
        print(f"Mean Absolute Error (MAE):     {metrics['mae']:.6f}")
        print(f"Mean Squared Error (MSE):      {metrics['mse']:.6f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
        print(f"R² Score:                      {metrics['r2_score']:.6f}")
        print(f"Median Absolute Error:         {metrics['median_ae']:.6f}")
        print("-"*60)
        print("Residual Statistics:")
        print(f"  Mean:                        {metrics['residual_mean']:.6f}")
        print(f"  Std Dev:                     {metrics['residual_std']:.6f}")
        print(f"  Min:                         {metrics['residual_min']:.6f}")
        print(f"  Max:                         {metrics['residual_max']:.6f}")
        print(f"  IQR:                         {metrics['residual_iqr']:.6f}")
        print("="*60 + "\n")


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                         cv: int = 5) -> Dict:
    """
    Perform k-fold cross validation.
    
    Args:
        model: Sklearn model or pipeline
        X: Features
        y: Target
        cv: Number of folds
        
    Returns:
        Dictionary with CV scores
    """
    # Compute different metrics via CV
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    neg_mae = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    neg_rmse = cross_val_score(
        model, X, y, cv=cv, 
        scoring='neg_root_mean_squared_error'
    )
    
    return {
        "r2_scores": r2_scores,
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
        "mae_scores": -neg_mae,
        "mae_mean": (-neg_mae).mean(),
        "mae_std": (-neg_mae).std(),
        "rmse_scores": -neg_rmse,
        "rmse_mean": (-neg_rmse).mean(),
        "rmse_std": (-neg_rmse).std(),
    }


def print_cv_report(cv_results: Dict) -> None:
    """
    Print formatted cross-validation report.
    
    Args:
        cv_results: Results from cross_validate_model
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION REPORT (5-fold)")
    print("="*60)
    print(f"R² Score:     {cv_results['r2_mean']:.6f} (+/- {cv_results['r2_std']:.6f})")
    print(f"MAE:          {cv_results['mae_mean']:.6f} (+/- {cv_results['mae_std']:.6f})")
    print(f"RMSE:         {cv_results['rmse_mean']:.6f} (+/- {cv_results['rmse_std']:.6f})")
    print("="*60 + "\n")
