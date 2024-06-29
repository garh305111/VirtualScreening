# Virtual Screening

This repository contains Python scripts for ML based preprocessing for virtual screening:

## Standard.py

This script provides functions for standardization, scaling, feature selection, and PCA preprocessing techniques.

### Functions Provided:

- **standard_preprocessing(data)**: Performs standardization (Z-score normalization) on the input data.
- **maxmin_preprocessing(data)**: Performs min-max scaling on the input data.
- **correlation_feature_selection(data, r_threshold)**: Performs feature selection based on correlation with the output variable.
- **decision_tree_feature_selection(data, threshold)**: Performs feature selection using a decision tree model.
- **PCA_preprocessing(data, com_number)**: Performs Principal Component Analysis (PCA) for dimensionality reduction.
- **RFE_feature_selection(data, n_features_to_select)**: Performs Recursive Feature Elimination (RFE) using a decision tree model.

### Example Usage:

```python
import pandas as pd
from Standard import *

# Load your data (assuming 'data' is your DataFrame)
df = pd.read_csv("./your_data.csv")
df = df.set_index('Name')

# Perform preprocessing pipeline
processed_data = preprocessing_pipeline(
    data=df, 
    pipeline=['standardization', 'correlation'], 
    r_threshold=0.3
)
print(processed_data)
