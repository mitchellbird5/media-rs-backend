import pandas as pd

from typing import List

def concat_string_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Concatenate multiple string columns of a DataFrame row-wise into a list of strings.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    columns : List[str]
        List of column names to concatenate.

    Returns
    -------
    List[str]
        List of concatenated strings, one per row.
    """
    # Check that all columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")

    # Ensure all columns are of string type
    for col in columns:
        if not pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Concatenate row-wise
    combined = df[columns].agg(' '.join, axis=1).tolist()
    return combined