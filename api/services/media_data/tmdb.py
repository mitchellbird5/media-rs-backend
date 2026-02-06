import os
import requests

from dataclasses import dataclass, field
from typing import Optional, List, Dict


from media_rs.utils.item_index import ItemIndex
from media_rs.utils.data_cache import get_data_cache

cache = get_data_cache()

TMDB_KEY = os.getenv("TMDB_KEY")
    
@dataclass
class MovieData:
    title: str = ""
    tmdb_id: Optional[int] = None
    imdb_id: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    genres: Dict[int, str] = field(default_factory=dict)
    overview: Optional[str] = None
    runtime: Optional[int] = None
    popularity: Optional[float] = None
    release_date: Optional[str] = None
    tagline: Optional[str] = None
    vote_average: Optional[float] = None

def get_movie_details(tmdb_id: int) -> MovieData:
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_KEY}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Use .get() for safety
    genres = {g["id"]: g["name"] for g in data.get("genres", []) if "id" in g and "name" in g}

    return MovieData(
        tmdb_id=tmdb_id,
        imdb_id=data.get("imdb_id"),
        poster_path=data.get("poster_path"),
        backdrop_path=data.get("backdrop_path"),
        genres=genres,
        overview=data.get("overview"),
        runtime=data.get("runtime"),
        popularity=data.get("popularity"),
        release_date=data.get("release_date"),
        tagline=data.get("tagline"),
        vote_average=data.get("vote_average")
    )

def get_movie_data(title: str) -> MovieData:
    item_idx = ItemIndex(cache.get("movies/item_index.pkl"))
    
    movie_id = item_idx.title_to_itemId[title]
    if movie_id is None:
        return MovieData(title=title)
    
    tmdb_id = item_idx.itemId_to_tmdbId.get(movie_id)

    if tmdb_id is None:
        return MovieData(title=title)
    
    data = get_movie_details(tmdb_id)
    data.title = title

    return data

def get_first_image_by_language(images_list: list, lang: str = "en") -> str | None:
    if not images_list:
        return None
    # Try to find an image with the given language
    for image in images_list:
        if image.get("iso_639_1") == lang:
            return image.get("file_path")
    # fallback to first image if none match language
    return images_list[0].get("file_path")

def get_multiple_movie_data(titles: List[str]) -> List[MovieData]:
    data: List[MovieData] = []
    for title in titles:
        try:
            data.append(get_movie_data(title))
        except Exception:
            data.append(MovieData(title=title))
    return data