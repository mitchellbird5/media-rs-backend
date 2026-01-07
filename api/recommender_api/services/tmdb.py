import os
import requests

from typing import List

from media_rs.utils.item_index import ItemIndex
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache

cache = get_movie_data_cache()

def get_movie_images(
    title: str
) -> dict:
    
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    idx = item_idx.title_to_idx[title]
    movie_id = item_idx.idx_to_movieId[idx]
    tmdb_id = item_idx.movieId_to_tmdbId[movie_id]
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/images"

    TMDB_KEY = os.getenv("TMDB_KEY")

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_KEY}"
    }

    response = requests.get(url, headers=headers)

    return response.json()

def get_multiple_movie_images(
    titles: List[str]
) -> dict:
    
    images = {}
    for title in titles:
        images[title] = get_movie_images(title)
    
    return images