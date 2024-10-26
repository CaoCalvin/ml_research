# preprocessing
from sklearn.preprocessing import StandardScaler
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

def scale(X: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame]:
    """Scales given feature matrix using StandardScaler

    Created: 2024/10/01

    Args:
        X (pd.DataFrame): feature matrix

    Returns:
        StandardScaler: scaler fit to model.
        pd.DataFrame: scaled version of feature matrix
    """

    # fit scaler automatically on feature matrix
    scaler = StandardScaler()
    scaler.fit(X)
    
    # use fit model to rescale X
    X_scaled = scaler.transform(X)

    return scaler, pd.DataFrame(X_scaled)

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
        X_CV
        y_CV
        X_test
        y_test
    """

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size = (1 - (CV_portion + test_portion)), random_state = seed)
    X_CV, X_test, y_CV, y_test = train_test_split(X_temp, y_temp, train_size = CV_portion / (CV_portion + test_portion), random_state = seed) 
    return X_train, y_train, X_CV, y_CV, X_test, y_test


def grid_search(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                param_grid: dict, train_model_callback):
    """Grid search using provided train_model_callback function

    Created: 2024/10/15

    Parameters:
    - X_train: features
    - y_train: labels
    - param_grid: dictionary where keys are parameter names (str) and values are lists of parameter values to search over
    - train_model_callback: function that trains a model and returns the trained model

    Returns:
    - pandas DataFrame where rows are combinations of parameter values and each cell contains a trained model
    """
    
    # Get all combinations of parameters from param_grid
    param_combinations = list(itertools.product(*param_grid.values()))
    
    # Create an empty DataFrame to store the models with param combinations as the index (using MultiIndex)
    index = pd.MultiIndex.from_tuples(param_combinations, names=param_grid.keys())
    results_df = pd.DataFrame(index=index, columns=['Model'])
    
    # Loop through each combination of parameters
    for params in param_combinations:
        # Create a dictionary of parameter names and their respective values for this combination
        param_dict = dict(zip(param_grid.keys(), params))
        
        # Train the model using the train_model_callback and the current parameter combination
        model = train_model_callback(X_train, y_train, **param_dict)
        
        # Store the model in the corresponding row
        results_df.loc[params, 'Model'] = model
    
    return results_df


