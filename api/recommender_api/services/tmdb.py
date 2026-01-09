import os
import requests

from typing import List

from media_rs.utils.item_index import ItemIndex
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache

cache = get_movie_data_cache()

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class MovieImageData:
    title: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None

def get_movie_images(title: str) -> MovieImageData:
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
    response.raise_for_status()  # raises exception for HTTP errors
    data = response.json()

    # Pick first poster and backdrop if available
    poster_path = get_first_image_by_language(data.get("posters"), lang="en")
    backdrop_path = get_first_image_by_language(data.get("backdrops"), lang="en")


    return MovieImageData(title=title, poster_path=poster_path, backdrop_path=backdrop_path)

def get_multiple_movie_images(titles: List[str]) -> List[MovieImageData]:
    images = []
    for title in titles:
        try:
            images.append(get_movie_images(title))
        except Exception:
            # if one movie fails, still continue with the rest
            images.append(MovieImageData(title=title))
    return images

def get_first_image_by_language(images_list: list, lang: str = "en") -> str | None:
    if not images_list:
        return None
    # Try to find an image with the given language
    for image in images_list:
        if image.get("iso_639_1") == lang:
            return image.get("file_path")
    # fallback to first image if none match language
    return images_list[0].get("file_path")
