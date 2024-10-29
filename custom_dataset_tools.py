 
import pandas as pd

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