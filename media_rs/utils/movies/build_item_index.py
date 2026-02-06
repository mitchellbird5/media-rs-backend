import pandas as pd

from typing import Dict, Any

def build_movie_item_index(
    movies: pd.DataFrame, 
    links: pd.DataFrame
) -> Dict[str, Any]:
    idx_to_itemId = dict(enumerate(movies["itemId"].values))
    itemId_to_idx = {mid: idx for idx, mid in idx_to_itemId.items()}

    itemId_to_title = dict(zip(movies["itemId"], movies["title"]))
    title_to_itemId = {title: mid for mid, title in itemId_to_title.items()}

    itemId_to_imdbId = dict(zip(links["itemId"], links["imdbId"]))
    itemId_to_tmdbId = dict(zip(links["itemId"], links["tmdbId"]))

    imdbId_to_itemId = {imdb: mid for mid, imdb in itemId_to_imdbId.items()}
    tmdbId_to_itemId = {tmdb: mid for mid, tmdb in itemId_to_tmdbId.items()}

    return {
        "num_items": len(movies),

        # index layer
        "idx_to_itemId": idx_to_itemId,
        "itemId_to_idx": itemId_to_idx,

        # presentation layer
        "itemId_to_title": itemId_to_title,
        "title_to_itemId": title_to_itemId,

        # external IDs
        "itemId_to_imdbId": itemId_to_imdbId,
        "itemId_to_tmdbId": itemId_to_tmdbId,
        "imdbId_to_itemId": imdbId_to_itemId,
        "tmdbId_to_itemId": tmdbId_to_itemId,
    }
