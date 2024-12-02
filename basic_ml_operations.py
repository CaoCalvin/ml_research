# preprocessing
from sklearn.model_selection import train_test_split

# data management
import ml_data_objects as mdo
import pandas as pd
import numpy as np

# ML 
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

# evaluation
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve

# other
import itertools

def train_XGB_reg(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> XGBRegressor:
    """Train an XGBoost Regressor.

    Created: 2024/11/16

    Args:
        X_train (pd.DataFrame): feature matrix
        y_train (np.ndarray): labels

    Returns:
        XGBRegressor: The trained model
    """

    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of rows"
    assert y_train.ndim == 1, "y_train must be a 1D array"

    # round all kwargs to nearest integer
    kwargs = {k: round(v) for k, v in kwargs.items()}

    model = XGBRegressor(**kwargs, objective='reg:squarederror', eval_metric='rmse')
    model.fit(X_train, y_train)
    return model

def train_XGB_clf(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> XGBClassifier:
    """Train an XGBoost Classifier.

    Created: 2024/11/30

    Args:
        X_train (pd.DataFrame): feature matrix
        y_train (np.ndarray): labels

    Returns:
        XGBClassifier: The trained model
    """

    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of rows"
    assert y_train.ndim == 1, "y_train must be a 1D array"

    # round all kwargs to nearest integer
    kwargs = {k: round(v) for k, v in kwargs.items()}

    model = XGBClassifier(**kwargs, objective='binary:logistic', eval_metric='logloss')
    model.fit(X_train, y_train)
    return model    

# trains a multi-output regressor SVM model.
def train_SVR(X_train: pd.DataFrame, y_train: np.ndarray, **kwargs) -> SVR:
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

def calculate_pearson_coefficients(X_grid: np.ndarray[pd.DataFrame], y_grid: np.ndarray[pd.DataFrame]) -> np.ndarray:
    """
    Calculate Pearson coefficients for the given data.

    Created: 2024/11/12

    Args:
        x_grid (np.ndarray): 2D numpy array of DataFrames for x values
        y_grid (np.ndarray): 2D numpy array of DataFrames for y values

    Returns:
        np.ndarray: 2D numpy array of Pearson coefficients
    """
    assert(X_grid.shape == y_grid.shape), f"x and y grids must have the same shape, but they have shapes {X_grid.shape} and {y_grid.shape}"

    assert(X_grid.ndim == 2 and y_grid.ndim == 2), f"x and y grids must be 2D, but they are {X_grid.ndim}D and {y_grid.ndim}D, with shapes {X_grid.shape} and {y_grid.shape}"

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            assert isinstance(X_grid[i, j], pd.DataFrame), f"Cell ({i}, {j}) in x_grid must be a DataFrame, but {type(X_grid[i, j])} was found"
            assert isinstance(y_grid[i, j], pd.DataFrame), f"Cell ({i}, {j}) in y_grid must be a DataFrame, but {type(y_grid[i, j])} was found"
    
            assert X_grid[i, j].shape[1] == 1, f"Cell ({i}, {j}) in x_grid must have only 1 column, but {X_grid[i, j].shape[1]} was found"
            assert y_grid[i, j].shape[1] == 1, f"Cell ({i}, {j}) in y_grid must have only 1 column, but {y_grid[i, j].shape[1]} was found"

    num_rows, num_cols = X_grid.shape
    pearson_coeffs = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            x_data = X_grid[i, j]
            y_data = y_grid[i, j]
            pearson_coef, _ = pearsonr(x_data, y_data)
            pearson_coeffs[i, j] = pearson_coef

    return pearson_coeffs

def find_optimal_threshold(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # Calculate sensitivity and specificity for each threshold
    sensitivity = tpr
    specificity = 1 - fpr
    
    # Calculate mean squared error between sensitivity and specificity
    mse = (sensitivity - specificity) ** 2
    
    # Find the threshold that minimizes MSE
    optimal_idx = np.argmin(mse)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold