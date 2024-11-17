# preprocessing
from sklearn.model_selection import train_test_split

# data management
import ml_data_objects as mdo
import pandas as pd
import numpy as np

# ML 
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from xgboost import XGBRegressor

# evaluation
from scipy.stats import pearsonr

# other
import itertools

# Define train_XGB function
def train_XGB(X_train: pd.DataFrame, y_train: np.ndarray, random_state: int = 42, **kwargs) -> XGBRegressor:
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

    model = XGBRegressor(random_state=random_state, **kwargs)
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

def split(X: pd.DataFrame, y: pd.DataFrame, CV_portion: float, test_portion: float, 
          seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, test, CV

    Created: 2024/09/10

    Args:
        X (pd.DataFrame): feature matrix
        y (pd.DataFrame): labels
        CV_portion (float): portion, out of 1, of data that should be CV set
        test_portion (float): portion, out of 1, of data that should be test set
        seed (int): seed for random operations

    Returns:
        pd.DataFrame: X_train 
        pd.DataFrame: y_train
        pd.DataFrame: X_CV
        pd.DataFrame: y_CV
        pd.DataFrame: X_test
        pd.DataFrame: y_test
    """

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size = (1 - (CV_portion + test_portion)), random_state = seed)
    X_CV, X_test, y_CV, y_test = train_test_split(X_temp, y_temp, train_size = CV_portion / (CV_portion + test_portion), random_state = seed) 

    

    return X_train, y_train, X_CV, y_CV, X_test, y_test

def grid_search(X_train: pd.DataFrame, y_train: pd.DataFrame, x_axis_params: mdo.AxisParams, y_axis_params: mdo.AxisParams,
                train_model_callback: callable, **kwargs) -> np.ndarray:
    """Grid search using provided train_model_callback function

    Created: 2024/10/15

    Args:
      X_train (pd.DataFrame): features
      y_train (pd.DataFrame): labels
      x_axis_params (AxisParams): parameters for the x-axis
      y_axis_params (AxisParams): parameters for the y-axis
      train_model_callback (function): function that trains a model and returns the trained model

    Returns:
      np.ndarray: 2D numpy array where each cell contains a trained model
    """

    # Get all combinations of parameters from x_axis_params and y_axis_params
    param_combinations = list(itertools.product(x_axis_params.values, y_axis_params.values))

    # Create a 2D numpy array to store the models
    num_rows = len(x_axis_params.values)
    num_cols = len(y_axis_params.values)
    model_grid = np.empty((num_rows, num_cols), dtype=object)

    # Loop through each combination of parameters
    for i, (x_param, y_param) in enumerate(param_combinations):
        # Train the model using the train_model_callback and the current parameter combination
        model = train_model_callback(X_train, y_train, **dict(zip([x_axis_params.name, y_axis_params.name], [x_param, y_param])), **kwargs)

        # Store the model in the corresponding row and column
        row_idx = i % num_rows
        col_idx = i // num_rows
        model_grid[row_idx, col_idx] = model

    return model_grid

def k_fold_grid_search(X_train_scaled: pd.DataFrame, y_train_scaled: pd.DataFrame, 
                       x_axis_params: mdo.AxisParams, y_axis_params: mdo.AxisParams,
                       train_model_callback: callable, kfold: KFold, **kwargs) -> np.ndarray:
    """Grid search using k-fold cross-validation and provided train_model_callback function

    Created: 2024/11/10

    Args:
      X_train_scaled (pd.DataFrame): features
      y_train_scaled (pd.DataFrame): labels
      x_axis_params (AxisParams): parameters for the x-axis
      y_axis_params (AxisParams): parameters for the y-axis
      train_model_callback (function): function that trains a model and returns the trained model
      kfold (KFold): KFold object for cross-validation
      **kwargs (dict): additional arguments to pass to train_model_callback

    Returns:
      np.ndarray: 3D numpy array where each cell contains a trained model
    """

    # Get all combinations of parameters from x_axis_params and y_axis_params
    param_combinations = list(itertools.product(x_axis_params.values, y_axis_params.values))

    # Create a 3D numpy array to store the models
    num_rows = len(x_axis_params.values)
    num_cols = len(y_axis_params.values)
    model_grid = np.empty((kfold.n_splits, num_rows, num_cols), dtype=object)

    # Perform k-fold cross-validation
    fold = 0
    for train_index, _ in kfold.split(X_train_scaled):
        X_fold = X_train_scaled.iloc[train_index]
        y_fold = y_train_scaled.iloc[train_index]
        
        # Loop through each combination of parameters
        for i, (x_param, y_param) in enumerate(param_combinations):
            # Train the model using the train_model_callback and the current parameter combination
            model = train_model_callback(X_fold, y_fold.values.ravel(), 
                                         **{x_axis_params.name: x_param, y_axis_params.name: y_param}, 
                                         **kwargs)
            model_grid[fold, i % num_rows, i // num_rows] = model
        fold += 1

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


    
def calculate_pearson_coefficients(x_grid: np.ndarray[pd.DataFrame], y_grid: np.ndarray[pd.DataFrame]) -> np.ndarray:
    """
    Calculate Pearson coefficients for the given data.

    Created: 2024/11/12

    Args:
        x_grid (np.ndarray): 2D numpy array of DataFrames for x values
        y_grid (np.ndarray): 2D numpy array of DataFrames for y values

    Returns:
        np.ndarray: 2D numpy array of Pearson coefficients
    """
    assert(x_grid.shape == y_grid.shape), f"x and y grids must have the same shape, but they have shapes {x_grid.shape} and {y_grid.shape}"
    num_rows, num_cols = x_grid.shape
    pearson_coeffs = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            x_data = x_grid[i, j]
            y_data = y_grid[i, j]
            pearson_coef, _ = pearsonr(x_data.iloc[:, 0], y_data.iloc[:, 0])
            pearson_coeffs[i, j] = pearson_coef

    return pearson_coeffs
