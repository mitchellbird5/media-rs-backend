import pandas as pd

from pathlib import Path
from typing import Tuple

from media_rs.utils.load_data import (
    load_dataframe_from_json
)

def load_all_book_data(
    wdir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads book data to train model

    Args:
        wdir (Path): Directory containing data

    Returns:
        _type_: _description_
    """
    
    books = load_dataframe_from_json(wdir.joinpath("metadata.json"))
    ratings = load_dataframe_from_json(wdir.joinpath("ratings.json"))
    tags = load_dataframe_from_json(wdir.joinpath("tags.json"))
    
    tags = tags.rename(columns={"id" : "itemId"})
    books = books.rename(columns={"item_id" : "itemId"})
    ratings = ratings.rename(columns={"user_id" : "userId", "item_id" : "itemId"})

    # books = add_tags_to_books(books, tags)
    
    return books, ratings, tags

def add_tags_to_books(
    books: pd.DataFrame, 
    tags: pd.DataFrame
) -> pd.DataFrame:
    # Group tags by id
    book_tags = tags.groupby('itemId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    # Merge with books
    books = books.merge(book_tags, on='itemId')
    books['tag'] = books['tag'].fillna('')
    return books