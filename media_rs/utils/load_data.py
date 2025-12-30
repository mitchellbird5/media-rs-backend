import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from typing import Tuple

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load CSV
    """
    df = pd.read_csv(csv_path).fillna('')
    return df

def save_numpy(array: np.ndarray, path: str):
    np.save(path, array)

def load_numpy(path: str) -> np.ndarray:
    return np.load(path)

def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_metadata(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)

def load_metadata(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
