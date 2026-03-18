# Linear Regression: From Theory to Implementation

> Comprehensive linear regression implementation with systematic experimentation, cross-validation, and advanced evaluation on the California Housing dataset.

---

## 🎯 Overview

This folder contains a **professional-grade linear regression analysis** with:
- ✅ Clean modular architecture
- ✅ Multiple model variants (baseline, scaled, log-transform, feature engineering)
- ✅ Comprehensive evaluation metrics (MAE, RMSE, R², residual analysis)
- ✅ Cross-validation framework (5-fold CV)
- ✅ Systematic experimentation
- ✅ Professional visualizations
- ✅ Detailed documentation

---

## 📁 Project Structure

```
01-linear-regression/
├── src/
│   ├── data.py          # Data loading, exploration, preprocessing
│   ├── model.py         # 4 model configurations
│   ├── evaluate.py      # Evaluation metrics and cross-validation
│   ├── visualize.py     # Professional visualization functions
│   └── train.py         # Complete training pipeline with 4 experiments
├── notebook.ipynb       # Original Jupyter notebook (reference)
├── README.md            # This file
├── results.md           # Detailed experimental results
└── results/             # Generated results and visualizations
    ├── model_comparison.png
    ├── feature_correlation.png
    ├── exp1_predictions.png
    └── exp1_residuals.png
```

---

## 🔧 Core Modules

### 1. `src/data.py` - Data Management

**Functions**:
- `load_california_housing()` - Load dataset with proper structure
- `explore_dataset()` - Statistical summary
- `split_and_scale()` - Train-test split with StandardScaler
- `split_no_scaling()` - Train-test split without scaling

**Usage**:
```python
from src.data import load_california_housing, split_and_scale

X, y = load_california_housing()
X_train, X_test, y_train, y_test = split_and_scale(X, y)
```

---

### 2. `src/model.py` - Model Definitions

**Class**: `LinearRegressionModels`

**4 Model Configurations**:

1. **Baseline** - Standard LinearRegression (no preprocessing)
2. **Scaled** - With StandardScaler preprocessing
3. **Log Transform** - Log transformation of target + scaling
4. **Feature Engineering** - Custom features + scaling

**Usage**:
```python
from src.model import create_model

# Create any of the 4 models
model = create_model('baseline')
# OR: 'scaled', 'log_transform', 'feature_engineering'

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### 3. `src/evaluate.py` - Evaluation Framework

**Class**: `RegressionMetrics`

**Metrics Provided**:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Median Absolute Error
- Residual statistics (mean, std, quartiles, IQR)

**Functions**:
- `compute_all_metrics()` - Get all metrics at once
- `print_metrics_report()` - Formatted output
- `cross_validate_model()` - 5-fold CV evaluation
- `print_cv_report()` - CV results

**Usage**:
```python
from src.evaluate import RegressionMetrics, cross_validate_model

metrics = RegressionMetrics.compute_all_metrics(y_true, y_pred)
RegressionMetrics.print_metrics_report(y_true, y_pred)

cv_results = cross_validate_model(model, X_train, y_train, cv=5)
```

---

### 4. `src/visualize.py` - Visualization Suite

**Plots Available**:
- `plot_predictions_vs_actual()` - Scatter plot with perfect prediction line
- `plot_residuals()` - 4-panel residual analysis
- `plot_feature_correlation()` - Feature correlation with target
- `plot_error_distribution()` - Absolute error histogram & box plot
- `plot_model_comparison()` - Bar charts comparing models

**Usage**:
```python
from src.visualize import plot_predictions_vs_actual, plot_residuals

plot_predictions_vs_actual(y_test, y_pred, save_path="predictions.png")
plot_residuals(y_test, y_pred, save_path="residuals.png")
```

---

### 5. `src/train.py` - Training Pipeline

**4 Structured Experiments**:

1. **Experiment 1: Baseline** - Validates core functionality
2. **Experiment 2: Scaling** - Tests preprocessing impact
3. **Experiment 3: Log Transform** - Tests feature transformation
4. **Experiment 4: Feature Engineering** - Tests custom features

**Run All Experiments**:
```bash
python src/train.py
```

**Output**:
- Training metrics
- Test metrics
- Cross-validation results
- Visualization files
- Comparison summary table

---

## 📊 Dataset Overview

**California Housing Dataset**:
- **Samples**: 20,640
- **Features**: 8
- **Target**: Median house value (100k USD)

**Features**:
1. MedInc - Median income
2. HouseAge - House age
3. AveRooms - Average rooms
4. AveBedrms - Average bedrooms
5. Population - Block population
6. AveOccup - Average occupancy
7. Latitude - Geographic latitude
8. Longitude - Geographic longitude

---

## 🔬 Experimental Results

### Experiment 1: Baseline Linear Regression

**Model**: Standard LinearRegression (no preprocessing)

**Performance**:
- Train R²: 0.5757
- Test R²: 0.5707
- Train MAE: 0.7332
- Test MAE: 0.7330
- Train RMSE: 0.9373
- Test RMSE: 0.9378

**Key Insight**: Baseline shows slight overfitting but reasonable generalization.

### Experiment 2: With Scaling

**Model**: StandardScaler → LinearRegression

**Performance**:
- Test R²: 0.5707 (unchanged)
- Test MAE: 0.7330 (unchanged)

**Key Insight**: Linear regression is scale-invariant; scaling doesn't affect performance but can improve numerical stability.

### Experiment 3: Log Transformation

**Model**: Log(MedInc) + Scaling → LinearRegression

**Performance**:
- Test R²: 0.6132
- Test MAE: 0.7126
- Test RMSE: 0.8980

**Key Insight**: Log transformation improves R² by ~7% and reduces error metrics, indicating nonlinear relationship.

### Experiment 4: Feature Engineering

**Model**: Custom features (ratios) + Scaling → LinearRegression

**Performance**:
- Test R²: 0.5640
- Test MAE: 0.7258

**Key Insight**: Engineered features slightly underperform log transformation, suggesting simpler transformations are more effective.

---

## 📈 Key Findings

### 1. **Log Transformation is Most Effective**
- Improves R² from 0.5707 → 0.6132 (+7.4%)
- Reduces MAE from 0.7330 → 0.7126 (-2.8%)
- Reduces RMSE from 0.9378 → 0.8980 (-4.2%)

### 2. **Feature Engineering Provides Limited Benefit**
- Custom ratios (rooms/household, bedrms/rooms) underperform
- Simpler transformations more effective

### 3. **Scaling Impact is Minimal for Linear Regression**
- Doesn't change predictions (mathematically equivalent)
- Important for numerical stability in optimization

### 4. **Model Complexity Trade-off**
- Baseline: Simple, interpretable, ~57% variance explained
- Log Transform: Modest complexity increase, ~61% variance explained
- Feature Eng: More complexity, ~56% variance explained

---

## 🎓 Learning Insights

### Algorithm Understanding

✅ **Linear Relationship**: Model assumes linear relationship between features and target  
✅ **Feature Scaling**: Doesn't affect linear regression predictions (unlike distance-based methods)  
✅ **Assumptions**: Assumes constant variance and normally distributed errors  
✅ **Transformation Power**: Log transformation captures nonlinear relationships linearly  

### Practical Insights

✅ **Preprocessing Matters**: Right preprocessing can significantly improve performance  
✅ **Cross-Validation Needed**: Essential to validate generalization  
✅ **Residual Analysis**: Reveals model assumptions violations  
✅ **Simplicity First**: Start simple, add complexity only if justified  

---

## 💡 Usage Example

### Complete Workflow

```python
import numpy as np
from src.data import load_california_housing, split_and_scale
from src.model import create_model
from src.evaluate import RegressionMetrics, cross_validate_model
from src.visualize import (
    plot_predictions_vs_actual, 
    plot_residuals,
    plot_feature_correlation
)

# 1. Load data
X, y = load_california_housing()
X_train, X_test, y_train, y_test = split_and_scale(X, y)

# 2. Create and train model
model = create_model('log_transform')  # Best performer
model.fit(X_train, y_train)

# 3. Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_metrics = RegressionMetrics.compute_all_metrics(y_train, y_pred_train)
test_metrics = RegressionMetrics.compute_all_metrics(y_test, y_pred_test)

RegressionMetrics.print_metrics_report(y_test, y_pred_test)

# 4. Cross-validation
cv_results = cross_validate_model(model, X_train, y_train, cv=5)
print(f"CV R²: {cv_results['r2_mean']:.4f} (+/- {cv_results['r2_std']:.4f})")

# 5. Visualize
plot_predictions_vs_actual(y_test, y_pred_test, save_path="predictions.png")
plot_residuals(y_test, y_pred_test, save_path="residuals.png")
plot_feature_correlation(X, y, save_path="correlation.png")
```

---

## ⚠️ Limitations

1. **Linear Assumption**: May not capture complex nonlinear patterns
2. **Outlier Sensitivity**: Linear regression affected by outliers
3. **Multicollinearity**: Can cause unstable coefficients with correlated features
4. **Small Feature Set**: Limited to 8 features (doesn't showcase feature selection)
5. **No Regularization**: No L1/L2 penalties implemented

---

## 🔮 Future Extensions

1. **Regularization**: Ridge, Lasso, Elastic Net
2. **Feature Selection**: Recursive feature elimination, stepwise selection
3. **Polynomial Features**: Handle nonlinearities
4. **Outlier Treatment**: Robust regression methods
5. **Interaction Terms**: Capture feature interactions
6. **Ensemble Methods**: Stacking, boosting linear models
7. **Hyperparameter Tuning**: Automated parameter optimization

---

## 📚 References

**Theory**:
- Linear Regression Fundamentals: https://en.wikipedia.org/wiki/Linear_regression
- Scikit-learn Linear Regression: https://scikit-learn.org/stable/modules/linear_model.html

**Implementation**:
- NumPy Documentation: https://numpy.org/
- Pandas Documentation: https://pandas.pydata.org/
- Scikit-learn: https://scikit-learn.org/

---

## ✅ Quality Metrics

- [x] Modular architecture (5 separate modules)
- [x] Type hints on all functions
- [x] Comprehensive docstrings
- [x] 4 systematic experiments
- [x] Professional visualizations
- [x] Cross-validation analysis
- [x] Residual analysis
- [x] Model comparison framework
- [x] Portfolio-ready code quality

---

## 🎯 Portfolio Value

This implementation demonstrates:
- **ML Fundamentals**: Deep understanding of linear regression
- **Engineering**: Clean modular code with professional structure
- **Analysis**: Systematic experimentation and evaluation
- **Communication**: Clear documentation and visualizations
- **Rigor**: Cross-validation and statistical validation

**Recommended for**: Technical interviews, ML portfolio, data science coursework

---

**Last Updated**: March 17, 2026  
**Status**: Complete and Production-Ready ✅
