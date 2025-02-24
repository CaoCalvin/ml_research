# Preprocessing
from sklearn.model_selection import train_test_split

# Data management
import ml_data_objects as mdo
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import KFold
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# Evaluation
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, roc_curve

# Other utilities
import itertools

def continuous_to_binary_quantile(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Converts continuous values in a DataFrame to binary (True/False) based on a quantile threshold.
    Each cell is "True" if its value is in the top specified percentile of its column.

    Created: 2024/10/02

    Args:
        df (pd.DataFrame): Input DataFrame with numeric values.
        threshold (float): Value between 0 and 1 specifying the quantile threshold.

    Returns:
        pd.DataFrame: DataFrame with boolean values indicating whether each value is in the top threshold.
    """
    # Initialize an empty DataFrame with the same shape and indices as the input
    top_df = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        # Compute the threshold value for the given quantile
        threshold_val = df[col].quantile(threshold)
        # Assign True if the value is greater than or equal to the threshold
        top_df[col] = df[col] >= threshold_val

    return top_df

def continuous_to_binary_absolute(predictions_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Converts continuous regression predictions to binary classifications based on an absolute threshold.

    Created: 2025/01/10

    Args:
        predictions_df (pd.DataFrame): DataFrame containing regression predictions.
        threshold (float or dict): Threshold(s) for binary classification.
                                   Can be a single float or a dictionary of column-specific thresholds.

    Returns:
        pd.DataFrame: DataFrame with binary (True/False) values.
    """
    if isinstance(threshold, dict):
        # Apply different thresholds per column
        return predictions_df.apply(lambda x: x >= threshold[x.name])
    else:
        # Apply the same threshold to all columns
        return predictions_df > threshold

def continuous_to_binary_absolute_grid(df_array: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applies binary classification conversion to each DataFrame within a NumPy array.

    Created: 2025/01/03

    Args:
        df_array (np.ndarray): Array containing pandas DataFrames.
        threshold (float): Value above which an entry is classified as "True."

    Returns:
        np.ndarray: Array of the same shape containing classified DataFrames.
    """
    # Initialize an output array of the same shape
    result = np.empty_like(df_array, dtype=object)
    
    # Apply the binary classification function to each DataFrame in the array
    for i in np.ndindex(df_array.shape):
        result[i] = continuous_to_binary_absolute(df_array[i], threshold)
    
    return result

def train_XGB_regressor(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> XGBRegressor:
    """
    Train an XGBoost Regressor model.

    Created: 2024/11/16

    Args:
        X_train (pd.DataFrame): Feature matrix.
        y_train (np.ndarray): Target labels.
        **kwargs: Additional arguments passed to XGBRegressor.

    Returns:
        XGBRegressor: Trained model.
    """
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of rows"
    assert y_train.ndim == 1, "y_train must be a 1D array"

    # Round all float keyword arguments to the nearest integer
    kwargs = {k: round(v) if isinstance(v, float) else v for k, v in kwargs.items()}

    model = XGBRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_XGB_classifier(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> XGBClassifier:
    """
    Train an XGBoost Classifier model.

    Created: 2024/11/16

    Args:
        X_train (pd.DataFrame): Feature matrix.
        y_train (np.ndarray): Target labels.
        **kwargs: Additional arguments passed to XGBClassifier.

    Returns:
        XGBClassifier: Trained model.
    """
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of rows"
    assert y_train.ndim == 1, "y_train must be a 1D array"

    # Round all float keyword arguments to the nearest integer
    kwargs = {k: round(v) if isinstance(v, float) else v for k, v in kwargs.items()}

    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def find_optimal_threshold_absolute(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """
    Finds the optimal decision threshold for a binary classifier using the ROC curve.
    Works with any numeric predictions by applying min-max normalization.

    Created: 2025/02/24

    Args:
        y_true (pd.DataFrame): True binary labels (0/1 values).
        y_pred (pd.DataFrame): Predicted values (continuous numeric values).

    Returns:
        float: The optimal threshold value in the original scale.
    """
    # Convert DataFrames to NumPy arrays
    y_true_arr = y_true.values.ravel()
    y_pred_arr = y_pred.values.ravel()
    
    # Ensure y_true contains only binary values
    if not np.array_equal(y_true_arr, y_true_arr.astype(bool)):
        raise ValueError("y_true must contain only binary values (0/1)")
    
    # Handle case where all predictions are identical
    pred_min, pred_max = np.min(y_pred_arr), np.max(y_pred_arr)
    if pred_max == pred_min:
        return float(pred_min)  # Return the single value as threshold
    
    # Normalize predictions
    y_pred_normalized = (y_pred_arr - pred_min) / (pred_max - pred_min)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_arr, y_pred_normalized)
    
    # Compute mean squared error between sensitivity and specificity
    mse = (tpr - (1 - fpr)) ** 2
    
    # Find the optimal threshold that minimizes MSE
    optimal_idx = np.argmin(mse)
    optimal_threshold_normalized = thresholds[optimal_idx]
    
    # Convert back to original scale
    optimal_threshold = optimal_threshold_normalized * (pred_max - pred_min) + pred_min
    
    return float(optimal_threshold)
