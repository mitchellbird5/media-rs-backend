import pandas as pd

from typing import List
from rs_types.model import ContentSimilarity, IdType

from utils.content_wrapper import get_id_from_similarity_result

def get_id_from_value(
    df: pd.DataFrame,
    keys: List[str],
    search_column: str,
    target_column: str,
) -> IdType:
    
    unique_vals = df[search_column].unique()
    
    for key in keys: 
        if key not in unique_vals:
            raise IndexError(f"Cannot find '{key}'")
    
    mask = df[search_column].isin(keys)
    id = df[mask][target_column].values
    
    return id


def get_value_from_id(
    df: pd.DataFrame,
    id: List[IdType],
    search_column: str,
) -> pd.DataFrame:
    mask = df[search_column].isin(id)
    return df[mask]


def get_result_from_similarity(
    df: pd.DataFrame,
    keys: List[IdType],
    search_column: str,
    result: List[ContentSimilarity]
) -> IdType:
    return get_value_from_id(
        df=df,
        id=get_id_from_similarity_result(result),
        search_column=search_column,
    )[keys]