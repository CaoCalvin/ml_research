 
import pandas as pd

def classify_top(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Returns dataframe of same shape as input dataframe, with each cell being "True" or "False" 
       depending on whether the original datapoint is in a specified top percentile of the original datapoints 
       in that column.

    Args:
        df (pd.DataFrame): input dataframe with numeric values
        threshold (float): number between 0 and 1 specifying "top" threshold.

    Returns:
        _type_: _description_
    """
    # create an empty dataframe
    top_df = pd.DataFrame(index=df.index, columns=df.columns)

    for col in df.columns:
        # Calculate the nth percentile value
        threshold_val = df[col].quantile(threshold)
        # Create a boolean mask where values are greater than or equal to the threshold
        top_df[col] = df[col] >= threshold_val

    return top_df

def assert_no_bad_values(df: pd.DataFrame):
    # Check for NaN values
    if df.isnull().values.any():
        raise ValueError("DataFrame contains NaN values.")
    
    # Check for non-numeric columns
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_columns:
        raise ValueError(f"DataFrame contains non-numeric columns: {non_numeric_columns}")
    
# returns the ID column and main dataframe of a matrix, assuming the training example IDs are in the first column.  
def seperate_ID(X: pd.DataFrame):
    return X.iloc[:, 0], X.iloc[:, 1:]

# returns a matrix where every interval rows are combined into a single row, with 
# each element being the average of the corresponding elements across the interval.
def avg_rows(df: pd.DataFrame, interval: int):
    # Number of new rows after taking average of every four rows
    new_num_rows = df.shape[0] // interval
    
    # Initialize output dataframe 
    df_avg = pd.DataFrame(columns=df.columns, index=range(new_num_rows), dtype=float)
    
    for i in range(new_num_rows):
        # calculate the average for the current group of rows
        avg_values = df.iloc[i * interval:(i + 1) * interval, :].mean(axis=0)
        
        # ensure the resulting averages are numeric
        df_avg.iloc[i, :] = avg_values.astype(float)

    return df_avg