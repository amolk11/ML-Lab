"""
Data loading and preprocessing for Linear Regression experiments.

This module handles:
- California Housing dataset loading
- Data exploration
- Train-test splitting
- Feature scaling and transformation
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


def load_california_housing() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load California Housing dataset.
    
    Returns:
        DataFrame with features, Series with target values
    """
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='Target')
    
    return X, y


def explore_dataset(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Explore dataset characteristics.
    
    Args:
        X: Features dataframe
        y: Target series
        
    Returns:
        Dictionary with exploration statistics
    """
    return {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "y_mean": y.mean(),
        "y_std": y.std(),
        "y_min": y.min(),
        "y_max": y.max(),
        "missing_values": X.isnull().sum().sum(),
    }


def split_and_scale(X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple:
    """
    Split data and apply scaling.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test (all scaled/split)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (
        pd.DataFrame(X_train_scaled, columns=X.columns),
        pd.DataFrame(X_test_scaled, columns=X.columns),
        y_train,
        y_test
    )


def split_no_scaling(X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple:
    """
    Split data without scaling (for experiments).
    
    Args:
        X: Features
        y: Target
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test (unscaled)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


if __name__ == "__main__":
    X, y = load_california_housing()
    print("Dataset loaded successfully!")
    print(f"Shape: {X.shape}")
    print(f"\nExploration stats:")
    stats = explore_dataset(X, y)
    for key, value in stats.items():
        print(f"  {key}: {value}")
