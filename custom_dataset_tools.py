 
import numpy as np
import pandas as pd

import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix

def classification_metrics(predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """
    Return neatly formatted table of f1 score, sensitivity, specificity, and kappa 
    for each column given two pandas boolean dataframes.

    Created 2024/10/29

    Parameters:
    predictions (pd.DataFrame): Predictions dataframe
    actual (pd.DataFrame): Actual dataframe
    """

    # Input validation
    if predictions.shape != actual.shape:
        raise ValueError("Predictions and actual dataframes must have the same shape")
    if not all(predictions.dtypes == bool) or not all(actual.dtypes == bool):
        raise ValueError("All columns in predictions and actual must be boolean")
    if predictions.isnull().values.any() or actual.isnull().values.any():
        raise ValueError("Predictions and actual dataframes must not contain missing values")

    metrics = []
    for i in range(len(predictions.columns)):
        y_pred = predictions.iloc[:, i]
        y_true = actual.iloc[:, i]
        
        # Calculate metrics
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Append to list
        metrics.append([f1, sensitivity, specificity, kappa])
    
    # print and return metrics
    metrics = pd.DataFrame(metrics, columns=['F1 Score', 'Sensitivity', 'Specificity', 'Kappa'])
    return metrics

def classify_top(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
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

def assert_no_bad_values(df: pd.DataFrame) -> None:
    """Raises Error if dataframe contains NaN or values or non-numeric type columns

    Created: 2024/10/02

    Args:
        df (pd.DataFrame): DataFrame to check

    Raises:
        ValueError: if contains NaN values
        ValueError: if contains non-numeric type columns
    """
    # Check for NaN values
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values.")
    
    # Check for non-numeric columns
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_columns:
        raise ValueError(f"DataFrame contains non-numeric columns: {non_numeric_columns}")
    
def first_col_to_row_labels(X: pd.DataFrame) -> pd.DataFrame:
    """Shaves the first column off of a matrix and sets it as the row index labels.  

    Created: 2024/10/01

    Args:
        X (pd.DataFrame): DataFrame with labels/IDs in first column

    Returns:
        pd.DataFrame: label DataFrame
        pd.DataFrame: other columns from original DataFrame
    """
    labels = X.iloc[:, 0]
    data = pd.DataFrame(X.iloc[:, 1:])
    data.index = labels

    # this dosesn't solve the problem
    # data.index = data.index.get_level_values(0)

    return data

def avg_rows(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    """returns a matrix where every [interval] rows are combined into a single row, with 
       each element being the average of the corresponding elements across the interval.

    Created: 2024/10/03

    Args:
        df (pd.DataFrame): a pandas dataframe
        interval (int): size of each group to average into one row

    Returns:
        pd.DataFrame: new dataframe with every [interval] rows averaged together.
    """
    # Number of new rows after taking average of every four rows
    new_num_rows = df.shape[0] // interval

    # Every 4th row label
    new_row_labels = df.index[::interval]
    
    # Initialize output dataframe 
    df_avg = pd.DataFrame(columns=df.columns, index=range(new_num_rows), dtype=float)
    
    for i in range(new_num_rows):
        # calculate the average for the current group of rows
        avg_values = df.iloc[i * interval:(i + 1) * interval, :].mean(axis=0)
        
        # ensure the resulting averages are numeric
        df_avg.iloc[i, :] = avg_values.astype(float)

    df_avg.index = new_row_labels

    return df_avg


def np_array_of_dfs(df: pd.DataFrame, shape: tuple) -> np.ndarray:
    """Create a numpy array where each cell contains a reference to the input dataframe

    Args:
        df (pd.DataFrame): input dataframe
        shape (tuple): shape of the numpy array (any number of dimensions)

    Returns:
        np.ndarray: numpy array where each cell contains a reference to the input dataframe
    """
    array = np.empty(shape, dtype=object)

    # Iterate through every cell in the numpy array
    for index, _ in np.ndenumerate(array):
        array[index] = df

    return array

def pretty_print_np_array_of_dfs(arr, rows_per_df=6):
    for i in range(arr.shape[0]):
        row_dfs = []
        max_height = 0
        
        for j in range(arr.shape[1]):
            df = arr[i, j]
            if len(df) > rows_per_df:
                df_str = pd.concat([df.head(rows_per_df // 2), pd.DataFrame([["..."] * df.shape[1]], index=["..."], columns=df.columns), df.tail(rows_per_df // 2)]).to_string()
            else:
                df_str = df.to_string()
            
            df_lines = df_str.split('\n')
            max_height = max(max_height, len(df_lines))
            row_dfs.append(df_lines)
        
        print(f"Row {i}:")
        for line in range(max_height):
            row_output = []
            for df_lines in row_dfs:
                if line < len(df_lines):
                    row_output.append(df_lines[line].ljust(40))
                else:
                    row_output.append(' ' * 40)
            print('    '.join(row_output))
        
        print("\n" + "-" * 200 + "\n")
