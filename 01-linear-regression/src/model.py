"""
Linear Regression model definitions and training utilities.

This module provides:
- Basic Linear Regression
- Linear Regression with log transformation
- Feature engineering variants
- Pipeline construction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from typing import Tuple


class LinearRegressionModels:
    """Container for different linear regression model configurations."""
    
    @staticmethod
    def baseline() -> LinearRegression:
        """
        Baseline Linear Regression model.
        
        Returns:
            Fitted LinearRegression instance
        """
        return LinearRegression()
    
    @staticmethod
    def with_scaling() -> Pipeline:
        """
        Linear Regression with feature scaling.
        
        Returns:
            Pipeline with StandardScaler + LinearRegression
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        return pipeline
    
    @staticmethod
    def with_log_transform() -> Pipeline:
        """
        Linear Regression with log transformation of target.
        
        Returns:
            Pipeline with preprocessing + LinearRegression
        """
        def log_transform(df):
            df_copy = df.copy()
            df_copy['MedInc'] = np.log1p(df_copy['MedInc'])
            return df_copy
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('log_income', FunctionTransformer(np.log1p, validate=False), ['MedInc']),
                ('scale', StandardScaler(), 
                 ['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 
                  'Latitude', 'Longitude'])
            ],
            remainder='passthrough'
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        return pipeline
    
    @staticmethod
    def with_feature_engineering() -> Pipeline:
        """
        Linear Regression with feature engineering.
        
        Returns:
            Pipeline with FE + preprocessing + LinearRegression
        """
        def feature_engineering(df):
            df = df.copy()
            df['AveBedrms_per_AveRooms'] = df['AveBedrms'] / df['AveRooms']
            df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
            return df.drop(['Population', 'AveOccup', 'Longitude', 'AveBedrms'], axis=1)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('log_income', FunctionTransformer(np.log1p, validate=False), ['MedInc']),
                ('scale', StandardScaler(),
                 ['HouseAge', 'AveRooms', 'AveBedrms_per_AveRooms', 'RoomsPerHousehold'])
            ],
            remainder='passthrough'
        )
        
        pipeline = Pipeline([
            ('feature_engineering', FunctionTransformer(feature_engineering, validate=False)),
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        return pipeline


def create_model(model_type: str = 'baseline') -> Pipeline:
    """
    Factory function to create model by name.
    
    Args:
        model_type: One of ['baseline', 'scaled', 'log_transform', 'feature_engineering']
        
    Returns:
        Model pipeline
    """
    models = {
        'baseline': LinearRegressionModels.baseline(),
        'scaled': LinearRegressionModels.with_scaling(),
        'log_transform': LinearRegressionModels.with_log_transform(),
        'feature_engineering': LinearRegressionModels.with_feature_engineering(),
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type]
