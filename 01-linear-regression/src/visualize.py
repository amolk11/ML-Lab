"""
Visualization functions for Linear Regression analysis.

This module provides:
- Residual plots
- Prediction vs actual plots
- Error distribution plots
- Feature correlation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               title: str = "Predictions vs Actual Values",
                               save_path: Optional[str] = None):
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = "Residual Analysis",
                   save_path: Optional[str] = None):
    """
    Plot residuals analysis.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='k', alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs Index
    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Index')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig


def plot_feature_correlation(X: pd.DataFrame, y: pd.Series,
                             title: str = "Feature Correlation with Target",
                             save_path: Optional[str] = None):
    """
    Plot feature correlation with target.
    
    Args:
        X: Features dataframe
        y: Target series
        title: Plot title
        save_path: Path to save figure
    """
    # Create correlation dataframe
    data = X.copy()
    data['Target'] = y
    
    correlations = data.corr()['Target'].drop('Target').sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if x < 0 else 'blue' for x in correlations]
    correlations.plot(kind='barh', ax=ax, color=colors, edgecolor='k', linewidth=0.5)
    
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                           title: str = "Prediction Error Distribution",
                           save_path: Optional[str] = None):
    """
    Plot absolute error distribution.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    errors = np.abs(y_true - y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='k', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Absolute Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Error Box Plot')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig


def plot_model_comparison(results_dict: dict,
                         title: str = "Model Performance Comparison",
                         save_path: Optional[str] = None):
    """
    Plot comparison of multiple models.
    
    Args:
        results_dict: Dictionary with model results (model_name -> metrics)
        title: Plot title
        save_path: Path to save figure
    """
    models = list(results_dict.keys())
    r2_scores = [results_dict[m]['r2_score'] for m in models]
    mae_scores = [results_dict[m]['mae'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² comparison
    axes[0].bar(models, r2_scores, edgecolor='k', alpha=0.7, color='skyblue')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² Score Comparison')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MAE comparison
    axes[1].bar(models, mae_scores, edgecolor='k', alpha=0.7, color='lightcoral')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('MAE Comparison')
    for i, v in enumerate(mae_scores):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    return fig
