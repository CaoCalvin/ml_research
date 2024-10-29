# preprocessing
from sklearn.model_selection import train_test_split

import pandas as pd

# ML models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

import itertools


# trains a multi-output regressor SVM model.
def train_SVR(X_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs) -> MultiOutputRegressor:
    """Train a Multioutput Regressor using SVM.

    Created: 2024/09/10

    Args:
        X_train (pd.DataFrame): feature matrix
        y_train (pd.DataFrame): labels

    Returns:
        MultiOutputRegressor: The model trained
    """

    svr = SVR(**kwargs)  
    model = MultiOutputRegressor(svr)
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


def grid_search(X_train: pd.DataFrame, y_train: pd.DataFrame, param_grid: dict, train_model_callback: callable, **kwargs) -> pd.DataFrame:
    """Grid search using provided train_model_callback function

    Created: 2024/10/15

    Args:
      X_train (pd.DataFrame): features
      y_train (pd.DataFrame): labels
      param_grid (dict): dictionary where keys are parameter names (str) and values are lists of parameter values to search over
      train_model_callback (function): function that trains a model and returns the trained model

    Returns:
      pd.DataFrame: rows are combinations of parameter values and each cell contains a trained model
    """
    
    # Get all combinations of parameters from param_grid
    param_combinations = list(itertools.product(*param_grid.values()))
    
    # Create a DataFrame to store the models with param combinations as the index (using MultiIndex)
    param_names = list(param_grid.keys())
    model_grid = pd.DataFrame(index=param_grid[param_names[0]], columns=param_grid[param_names[1]])
    
    # Loop through each combination of parameters
    for params in param_combinations:
        # Train the model using the train_model_callback and the current parameter combination
        model = train_model_callback(X_train, y_train, **dict(zip(param_names, params)), **kwargs)
        
        # Store the model in the corresponding row and column
        model_grid.loc[params[0], params[1]] = model
    
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