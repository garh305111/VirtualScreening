# Virtual Screening

This repository contains Python scripts for ML based preprocessing for virtual screening:

## Standard.py

Standardize.py provides functions for standardization, scaling, feature selection, and PCA preprocessing techniques.

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
```

## KNN_impute.py

KNN_impute.py is intended for K-Nearest Neighbors (KNN) imputation of missing values in datasets.

### Functions Provided:

-- **KNN_DataFill(data)**: implements KNN for filling missing values in a dataset.

#### Usage:

```python
import pandas as pd
from KNN_DataFill import KNN_DataFill

# Load your data
df = pd.read_csv("./your_data.csv")
df = df.set_index('Name')

# Example usage for filling missing values
missing_value_per_col = round(len(df.columns) * 0.05)
col_dict = {}

for col in df.columns:
    inter_df = intersection_cols(df.drop(columns=[col]))
    inter_df[col] = df[col]
    add_missing_df = add_missing(inter_df, col, missing_value_per_col)
    
    predictor = KNN_DataFill(
        add_missing_df, 
        col, 
        set_PCA_dist=True, 
        threshold=0.6,
        n_components=500
    )
    predictor.fit(k=10)
    filled_df = predictor.predict()
    rmse = predictor.evaluation()
    col_dict[col] = rmse

print(col_dict)
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
