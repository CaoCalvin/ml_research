import pandas as pd

def intersection(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the intersection of two DataFrames, preserving duplicates according to the minimum
    count of occurrences in either DataFrame.
    
    Created: 2024/03/03
    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
    Returns:
        pd.DataFrame: Intersection of the two DataFrames with controlled duplicates.
    Example:
        If a row appears 3 times in df1 and 2 times in df2, it will appear 2 times in the result.
    """
    # Get the initial intersection
    merged = pd.merge(df1, df2, how='inner')
    
    # Convert DataFrames to string representation of rows for counting
    df1_rows = df1.astype(str).agg('-'.join, axis=1)
    df2_rows = df2.astype(str).agg('-'.join, axis=1)
    merged_rows = merged.astype(str).agg('-'.join, axis=1)
    
    # Count occurrences in each DataFrame
    df1_counts = df1_rows.value_counts()
    df2_counts = df2_rows.value_counts()
    
    # For each unique row in the merged result, keep minimum count
    result_rows = []
    for row_str, count in merged_rows.value_counts().items():
        min_count = min(df1_counts[row_str], df2_counts[row_str])
        row_df = merged[merged.astype(str).agg('-'.join, axis=1) == row_str].head(min_count)
        result_rows.append(row_df)
    
    return pd.concat(result_rows) if result_rows else pd.DataFrame(columns=merged.columns)

def difference(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the difference between two DataFrames, handling duplicates.
    If a row appears a times in df1 and b times in df2, it will appear max(0, a-b) times in the result.
    
    Created: 2024/03/03
    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
    Returns:
        pd.DataFrame: Difference between the two DataFrames with controlled duplicates
    Example:
        If a row appears 5 times in df1 and 2 times in df2, it will appear 3 times in the result.
    """
    # Convert DataFrames to string representation of rows for counting
    df1_rows = df1.astype(str).agg('-'.join, axis=1)
    df2_rows = df2.astype(str).agg('-'.join, axis=1)
    
    # Count occurrences in each DataFrame
    df1_counts = df1_rows.value_counts()
    df2_counts = df2_rows.value_counts()
    
    # Calculate differences and keep only positive differences
    diff_counts = {}
    for row_str in df1_counts.index:
        count_diff = df1_counts[row_str] - df2_counts.get(row_str, 0)
        if count_diff > 0:
            diff_counts[row_str] = count_diff
    
    # Build result DataFrame
    result_rows = []
    for row_str, count in diff_counts.items():
        row_df = df1[df1_rows == row_str].head(count)
        result_rows.append(row_df)
    
    return pd.concat(result_rows) if result_rows else pd.DataFrame(columns=df1.columns)