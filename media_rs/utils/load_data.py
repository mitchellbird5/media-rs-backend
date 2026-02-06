import pandas as pd
import numpy as np
import pickle
import faiss

import re
import unicodedata

def norm(title: str) -> str:
    if not title:
        return ""

    # Unicode normalization (é → é)
    title = unicodedata.normalize("NFKC", title)

    # Lowercase
    title = title.lower()

    # Strip surrounding whitespace
    title = title.strip()

    # Replace common punctuation with spaces
    title = re.sub(r"[^\w\s()\-]", " ", title)
    title = re.sub(r"[()]", "", title)

    # Collapse whitespace
    title = re.sub(r"\s+", " ", title)

    return title

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load CSV
    """
    df = pd.read_csv(csv_path).fillna('')
    return df

def load_dataframe_from_json(json_path: str) -> pd.DataFrame:
    """
    Load JSON
    """
    df = pd.read_json(json_path, lines=True).fillna('')
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

def save_faiss_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)