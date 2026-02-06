import pandas as pd
from media_rs.utils.load_data import norm

def build_content_column(books: pd.DataFrame) -> pd.DataFrame:
    """
    Builds column of text content to use in embedding

    Args:
        books (pd.DataFrame): Dataframe of movie data

    Returns:
        pd.DataFrame: Updated dataframe with content column
    """
    books = books.sort_values("itemId").reset_index(drop=True)
    books["content"] = books[
        [
            "title", 
            "authors", 
            "description", 
            # "tag"
        ]
    ].fillna("").agg(" ".join, axis=1).apply(norm)
    return books
