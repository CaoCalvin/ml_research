# preprocessing
from sklearn.model_selection import train_test_split

# data management
import ml_data_objects as mdo
import pandas as pd
import numpy as np

# ML 
from sklearn.model_selection import KFold
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

# evaluation
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, roc_curve

# other
import itertools

def continuous_to_binary_quantile(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Returns dataframe of same shape as input dataframe, with each cell being "True" or "False" 
       depending on whether the original datapoint is in a specified top percentile of the original datapoints 
       in that column.

       Created: 2024/10/02

    Args:
        df (pd.DataFrame): input dataframe with numeric values
        threshold (float): number between 0 and 1 specifying "top" threshold.

    Returns:
        pd.DataFrame: new dataframe where each cell is True or False depending on whether the corresponding value in
                      the original dataframe is a "top" value
    """
    # create an empty dataframe
    top_df = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        # Calculate the nth percentile value
        threshold_val = df[col].quantile(threshold)
        # Create a boolean mask where values are greater than or equal to the threshold
        top_df[col] = df[col] >= threshold_val

    return top_df

def continuous_to_binary_absolute(predictions_df : pd.DataFrame, threshold : float) -> pd.DataFrame:
    """
    Convert regression predictions to binary classifications with boolean output

    Created: 2025/01/10
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with regression predictions
        threshold (float or dict): Threshold(s) for binary conversion
                                 Can be single float or dict of thresholds per column
    
    Returns:
        pd.DataFrame: Binary predictions (True/False)
    """
    if isinstance(threshold, dict):
        # Apply different thresholds per column
        return predictions_df.apply(lambda x: x >= threshold[x.name])
    else:
        # Apply same threshold to all columns
        return predictions_df > threshold

def continuous_to_binary_absolute_grid(df_array: np.ndarray, threshold: float) -> np.ndarray:
    """Applies classify_top to each dataframe in a numpy array.

    Created: 2025/01/03
    
    Args:
        df_array (np.ndarray): Array containing pandas DataFrames
        threshold (float): number above which a line is considered "top"
        
    Returns:
        np.ndarray: Array of same shape as input, containing classified DataFrames
    """
    # Create output array of same shape
    result = np.empty_like(df_array, dtype=object)
    
    # Iterate through array using ndindex to handle arbitrary dimensions
    for i in np.ndindex(df_array.shape):
        result[i] = continuous_to_binary_absolute(df_array[i], threshold)
        
    return result

def train_XGB_regressor(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> XGBRegressor:
    """Train an XGBoost Regressor.

    Created: 2024/11/16

    Args:
        X_train (pd.DataFrame): feature matrix
        y_train (np.ndarray): labels
        **kwargs: keyword arguments to pass to XGBRegressor. For regression pass {objective='reg:squarederror', eval_metric='rmse'}. 
                For classification pass {objective='binary:logistic' eval_metric='logloss'}

    Returns:
        XGBRegressor: The trained model
    """

    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of rows"
    assert y_train.ndim == 1, "y_train must be a 1D array"

    # round all float kwargs to nearest integer
    kwargs = {k: round(v) if isinstance(v, float) else v for k, v in kwargs.items()}

    model = XGBRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_XGB_classifier(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> XGBClassifier:
    """Train an XGBoost Classifier.

    Created: 2024/11/16

    Args:
        X_train (pd.DataFrame): feature matrix
        y_train (np.ndarray): labels
        **kwargs: keyword arguments to pass to XGBClassifier. For regression pass {objective='reg:squarederror', eval_metric='rmse'}. 
                For classification pass {objective='binary:logistic' eval_metric='logloss'}

    Returns:
        XGBClassifier: The trained model
    """

    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of rows"
    assert y_train.ndim == 1, "y_train must be a 1D array"

    # round all float kwargs to nearest integer
    kwargs = {k: round(v) if isinstance(v, float) else v for k, v in kwargs.items()}

    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

# trains a multi-output regressor SVM model.
def train_SVM_regressor(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> SVR:
    """Train an SVR Regressor using SVM.

    Created: 2024/09/10

    Args:
        X_train (pd.DataFrame): feature matrix
        y_train (np.ndarray): labels

    Returns:
        SVR: The model trained
    """

    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of rows"
    assert y_train.ndim == 1, "y_train must be a 1D array"

    model = SVR(**kwargs)  
    model.fit(X_train, y_train)
    return model

# trains a multi-output classifier SVM model.
def train_SVM_classifier(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> SVC:
    """Train an SVC Classifier using SVM.

    Created: 2024/09/10

    Args:
        X_train (pd.DataFrame): feature matrix
        y_train (np.ndarray): labels

    Returns:
        SVC: The model trained
    """

    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of rows"
    assert y_train.ndim == 1, "y_train must be a 1D array"

    model = SVC(**kwargs)  
    model.fit(X_train, y_train)
    return model




def train_model_grid(X_train: pd.DataFrame, y_train: pd.DataFrame, axis1_params: mdo.AxisParams, axis2_params: mdo.AxisParams,
                train_model_callback: callable, **kwargs) -> np.ndarray:
    """Grid search using provided train_model_callback function

    Created: 2024/10/15

    Args:
      X_train (pd.DataFrame): features
      y_train (pd.DataFrame): labels
      axis1_params (AxisParams): parameters for the first axis
      axis2_params (AxisParams): parameters for the second axis
      train_model_callback (function): function that trains a model and returns the trained model

    Returns:
      np.ndarray: 2D numpy array where each cell contains a trained model
    """

    # Get all combinations of parameters from axis1_params and axis2_params
    param_combinations = list(itertools.product(axis1_params.values, axis2_params.values))

    # Create a 2D numpy array to store the models
    num_rows = len(axis1_params.values)
    num_cols = len(axis2_params.values)
    model_grid = np.empty((num_rows, num_cols), dtype=object)

    # Loop through each combination of parameters
    for i, (param1, param2) in enumerate(param_combinations):
        # Train the model using the train_model_callback and the current parameter combination
        model = train_model_callback(X_train, np.ravel(y_train), **{axis1_params.name: param1, axis2_params.name: param2}, **kwargs)

        # Store the model in the corresponding row and column
        row_idx = i // num_cols
        col_idx = i % num_cols
        model_grid[row_idx, col_idx] = model

    return model_grid

def power_list(base: float, start: int, end: int) -> list:
    """Returns list of powers of base between two specified degrees, inclusive; used to generate grid search values

    Created: 2024/10/27

    Args:
        base (float): Base of exponents
        start (int): Exponent at start of list
        end (int): Exponent at end of list

    Returns:
        list: List of floats
    """
    assert start < end, "Start must be less than end"
    return [base**i for i in range(start, end + 1)]

def grid_predict(X_sc: pd.DataFrame, model_grid: np.ndarray) -> np.ndarray:
    """Evaluates single fold's grid search results using scaled test folds.

    Created: 2024/11/03

    Args:
        X_sc (pd.DataFrame): Scaled feature set.
        y_sc (pd.DataFrame): Scaled label set.
        model_grid (np.ndarray): 2D numpy array of models to be evaluated

    Returns:
        np.ndarray: 2D numpy array where each cell contains single-column dataframe of predictions 
    """
    # Initialize numpy array to store predictions for each parameter combination
    num_rows, num_cols = model_grid.shape
    y_pred_grid = np.empty((num_rows, num_cols), dtype=object)

    # Make predictions using the model in each cell
    for i in range(num_rows):
        for j in range(num_cols):
            model = model_grid[i, j]
            y_pred_grid[i, j] = pd.DataFrame(model.predict(X_sc))
    return y_pred_grid

def grid_predict_proba(X_sc: pd.DataFrame, model_grid: np.ndarray, target: int) -> np.ndarray:
    """Evaluates single fold's grid search results using scaled test folds, using model's predict_proba method.

    Created: 2024/11/03

    Args:
        X_sc (pd.DataFrame): Scaled feature set.
        y_sc (pd.DataFrame): Scaled label set.
        model_grid (np.ndarray): 2D numpy array of models to be evaluated
        target: (int): Which class to predict probabilities for (0 = first, etc.)

    Returns:
        np.ndarray: 2D numpy array where each cell contains single-column dataframe of predictions 
    """
    # Initialize numpy array to store predictions for each parameter combination
    num_rows, num_cols = model_grid.shape
    y_pred_grid = np.empty((num_rows, num_cols), dtype=object)

    # Make predictions using the model in each cell
    for i in range(num_rows):
        for j in range(num_cols):
            model = model_grid[i, j]
            y_pred_grid[i, j] = pd.DataFrame((model.predict_proba(X_sc))[:, target])
    return y_pred_grid

def calculate_pearson_coefficients(preds_grid: np.ndarray[pd.DataFrame], actuals_grid: np.ndarray[pd.DataFrame]) -> np.ndarray:
    """
    Calculate Pearson coefficients for the given data.

    Created: 2024/11/12

    Args:
        preds_grid (np.ndarray): 2D numpy array of DataFrames for predicted values
        actuals_grid (np.ndarray): 2D numpy array of DataFrames for actual values

    Returns:
        np.ndarray: 2D numpy array of Pearson coefficients
    """
    assert(preds_grid.shape == actuals_grid.shape), f"preds_grid and actuals_grid grids must have the same shape, but they have shapes {preds_grid.shape} and {actuals_grid.shape}"

    assert(preds_grid.ndim == 2 and actuals_grid.ndim == 2), f"preds_grid and actuals_grid grids must be 2D, but they are {preds_grid.ndim}D and {actuals_grid.ndim}D, with shapes {preds_grid.shape} and {actuals_grid.shape}"

    for i in range(preds_grid.shape[0]):
        for j in range(preds_grid.shape[1]):
            assert isinstance(preds_grid[i, j], pd.DataFrame), f"Cell ({i}, {j}) in preds_grid must be a DataFrame, but {type(preds_grid[i, j])} was found"
            assert isinstance(actuals_grid[i, j], pd.DataFrame), f"Cell ({i}, {j}) in actuals_grid must be a DataFrame, but {type(actuals_grid[i, j])} was found"
    
            assert preds_grid[i, j].shape[1] == 1, f"Cell ({i}, {j}) in preds_grid must have only 1 column, but {preds_grid[i, j].shape[1]} was found"
            assert actuals_grid[i, j].shape[1] == 1, f"Cell ({i}, {j}) in actuals_grid must have only 1 column, but {actuals_grid[i, j].shape[1]} was found"

    num_rows, num_cols = preds_grid.shape
    pearson_coeffs = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            x_data = preds_grid[i, j]
            y_data = actuals_grid[i, j]
            pearson_coef, _ = pearsonr(x_data, y_data)
            pearson_coeffs[i, j] = pearson_coef

    return pearson_coeffs

def calculate_f1_scores(preds_grid: np.ndarray[pd.DataFrame], actuals_grid: np.ndarray[pd.DataFrame]) -> np.ndarray:
    """
    Calculate F1 scores for binary classification for the given data.

    Created: 2024/11/12

    Args:
        preds_grid (np.ndarray): 2D numpy array of DataFrames for predicted values
        actuals_grid (np.ndarray): 2D numpy array of DataFrames for actual values

    Returns:
        np.ndarray: 2D numpy array of F1 scores
    """

    assert(preds_grid.shape == actuals_grid.shape), f"preds_grid and actuals_grid grids must have the same shape, but they have shapes {preds_grid.shape} and {actuals_grid.shape}"

    assert(preds_grid.ndim == 2 and actuals_grid.ndim == 2), f"preds_grid and actuals_grid grids must be 2D, but they are {preds_grid.ndim}D and {actuals_grid.ndim}D, with shapes {preds_grid.shape} and {actuals_grid.shape}"

    for i in range(preds_grid.shape[0]):
        for j in range(preds_grid.shape[1]):
            assert isinstance(preds_grid[i, j], pd.DataFrame), f"Cell ({i}, {j}) in preds_grid must be a DataFrame, but {type(preds_grid[i, j])} was found"
            assert isinstance(actuals_grid[i, j], pd.DataFrame), f"Cell ({i}, {j}) in actuals_grid must be a DataFrame, but {type(actuals_grid[i, j])} was found"
    
            assert preds_grid[i, j].shape[1] == 1, f"Cell ({i}, {j}) in preds_grid must have only 1 column, but {preds_grid[i, j].shape[1]} was found"
            assert actuals_grid[i, j].shape[1] == 1, f"Cell ({i}, {j}) in actuals_grid must have only 1 column, but {actuals_grid[i, j].shape[1]} was found"

    num_rows, num_cols = preds_grid.shape
    f1_scores = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            y_pred = preds_grid[i, j].squeeze()
            y_true = actuals_grid[i, j].squeeze()
            f1_scores[i, j] = f1_score(y_true, y_pred, zero_division=0)

    return f1_scores

def find_optimal_threshold_absolute(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """
    Find the optimal threshold for a binary classifier based on the ROC curve.
    Works with any numeric prediction values by applying min-max normalization.

    Parameters:
    y_true (pd.DataFrame): True binary labels (0/1 values)
    y_pred (pd.DataFrame): Predicted values (any numeric values)

    Returns:
    float: The optimal threshold value in the original scale
    """
    # Convert DataFrames to arrays and validate
    y_true_arr = y_true.values.ravel()
    y_pred_arr = y_pred.values.ravel()
    
    if not np.array_equal(y_true_arr, y_true_arr.astype(bool)):
        raise ValueError("y_true must contain only binary values (0/1)")
    
    # Handle case where all predictions are identical
    pred_min = np.min(y_pred_arr)
    pred_max = np.max(y_pred_arr)
    if pred_max == pred_min:
        return float(pred_min)  # Return the single value as threshold
    
    y_pred_normalized = (y_pred_arr - pred_min) / (pred_max - pred_min)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_arr, y_pred_normalized)
    
    # Calculate sensitivity and specificity
    sensitivity = tpr
    specificity = 1 - fpr
    
    # Calculate mean squared error between sensitivity and specificity
    mse = (sensitivity - specificity) ** 2
    
    # Find the normalized threshold that minimizes MSE
    optimal_idx = np.argmin(mse)
    optimal_threshold_normalized = thresholds[optimal_idx]
    
    # Denormalize threshold to original scale
    optimal_threshold = optimal_threshold_normalized * (pred_max - pred_min) + pred_min
    
    return float(optimal_threshold)