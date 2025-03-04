import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

def classification_metrics(predictions: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and return a table of classification metrics: F1 score, sensitivity, specificity, precision, and Cohen's kappa
    for each column in the given boolean dataframes.

    Created: 2024/10/29

    Parameters:
    predictions (pd.DataFrame): Boolean DataFrame containing predicted values.
    actual (pd.DataFrame): Boolean DataFrame containing actual values.

    Returns:
    pd.DataFrame: A DataFrame containing calculated metrics for each column.
    """

    # Ensure input DataFrames have the same shape
    if predictions.shape != actual.shape:
        raise ValueError("Predictions and actual dataframes must have the same shape")
    
    # Ensure all values are boolean
    if not all(predictions.dtypes == bool) or not all(actual.dtypes == bool):
        raise ValueError("All columns in predictions and actual must be boolean")
    
    # Ensure no missing values exist
    if predictions.isnull().values.any() or actual.isnull().values.any():
        raise ValueError("Predictions and actual dataframes must not contain missing values")
    
    metrics = []
    
    for i in range(len(predictions.columns)):
        y_pred = predictions.iloc[:, i]
        y_true = actual.iloc[:, i]
        
        # Handle degenerate cases where there is only one class in y_true or y_pred
        if len(np.unique(y_true)) == 1 or len(np.unique(y_pred)) == 1:
            metrics.append([0, 0, 0, 0, 0])  # Return zeros for invalid cases
            continue
        
        # Compute confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Compute individual metrics
        sensitivity = tp / (tp + fn)  # Also known as recall
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        metrics.append([f1, sensitivity, specificity, precision, kappa])
    
    # Create DataFrame from collected metrics
    return pd.DataFrame(metrics, 
                        columns=['F1 Score', 'Sensitivity', 'Specificity', 'Precision', 'Kappa'])

def assert_no_bad_values(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame contains no NaN values and only numeric columns.

    Created: 2024/10/02

    Parameters:
    df (pd.DataFrame): DataFrame to check.

    Raises:
    ValueError: If NaN values are present or if any non-numeric columns exist.
    """
    
    # Check for NaN values
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values.")
    
    # Identify non-numeric columns
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_columns:
        raise ValueError(f"DataFrame contains non-numeric columns: {non_numeric_columns}")

def first_col_to_row_labels(X: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the first column from a DataFrame and sets it as the row index.

    Created: 2024/10/01

    Parameters:
    X (pd.DataFrame): DataFrame with labels/IDs in the first column.

    Returns:
    pd.DataFrame: DataFrame with the first column set as row labels.
    """
    labels = X.iloc[:, 0]  # Extract first column as labels
    data = pd.DataFrame(X.iloc[:, 1:])  # Extract remaining columns
    data.index = labels  # Assign extracted labels as row index
    return data

def avg_rows(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    """
    Groups every [interval] rows and replaces them with their average values.

    Created: 2024/10/03

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    interval (int): Number of rows to group together and average.

    Returns:
    pd.DataFrame: New DataFrame where each [interval] rows are averaged into a single row.
    """
    new_num_rows = df.shape[0] // interval  # Calculate number of new rows
    df_avg = pd.DataFrame(columns=df.columns, index=range(new_num_rows), dtype=float)
    
    for i in range(new_num_rows):
        avg_values = df.iloc[i * interval:(i + 1) * interval, :].mean(axis=0)  # Compute mean per group
        df_avg.iloc[i, :] = avg_values.astype(float)  # Assign numeric values
    
    df_avg.index = range(new_num_rows)  # Reset index
    return df_avg

def np_array_of_dfs(df: pd.DataFrame, shape: tuple) -> np.ndarray:
    """
    Create a NumPy array where each cell contains a reference to the input DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    shape (tuple): Desired shape of the output NumPy array.

    Returns:
    np.ndarray: NumPy array where each element contains a reference to the input DataFrame.
    """
    array = np.empty(shape, dtype=object)  # Create empty array with specified shape
    
    for index, _ in np.ndenumerate(array):
        array[index] = df  # Fill array with DataFrame references
    
    return array

def pretty_print_np_array_of_dfs(arr, rows_per_df=6):
    """
    Print a NumPy array of DataFrames in a formatted manner, truncating long DataFrames.

    Parameters:
    arr (np.ndarray): NumPy array containing DataFrames.
    rows_per_df (int, optional): Maximum number of rows to display per DataFrame. Default is 6.
    """
    for i in range(arr.shape[0]):
        row_dfs = []
        max_height = 0
        
        for j in range(arr.shape[1]):
            df = arr[i, j]
            
            # Truncate long DataFrames
            if len(df) > rows_per_df:
                df_str = pd.concat([df.head(rows_per_df // 2), pd.DataFrame([['...'] * df.shape[1]], index=['...'], columns=df.columns), df.tail(rows_per_df // 2)]).to_string()
            else:
                df_str = df.to_string()
            
            df_lines = df_str.split('\n')
            max_height = max(max_height, len(df_lines))
            row_dfs.append(df_lines)
        
        print(f"Row {i}:")
        for line in range(max_height):
            row_output = []
            for df_lines in row_dfs:
                row_output.append(df_lines[line].ljust(40) if line < len(df_lines) else ' ' * 40)
            print('    '.join(row_output))
        
        print("\n" + "-" * 200 + "\n")

def random_subset(features: pd.DataFrame, labels: pd.DataFrame, p: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects a random subset of features and labels by keeping a proportion 'p' of samples.

    Created: 3/3/2025

    Parameters:
    - features (pd.DataFrame): Input features DataFrame
    - labels (pd.DataFrame): Input labels DataFrame
    - p (float): Proportion of rows to keep (0 = remove all, 1 = keep all)
    - random_state (int): Random seed for reproducibility

    Returns:
    - tuple[pd.DataFrame, pd.DataFrame]: Tuple of (subsampled features, subsampled labels)
    """
    assert 0 <= p <= 1, "Proportion p must be between 0 and 1"
    assert len(features) == len(labels), "Features and labels must have same number of rows"

    # Calculate the number of rows to keep
    n_keep = int(len(features) * p)

    if n_keep == 0:
        return pd.DataFrame(columns=features.columns), pd.DataFrame(columns=labels.columns)

    # Generate random indices
    indices = np.random.RandomState(random_state).choice(len(features), n_keep, replace=False)
    
    # Sample both DataFrames using same indices
    features_subset = features.iloc[indices]
    labels_subset = labels.iloc[indices]

    return features_subset, labels_subset
