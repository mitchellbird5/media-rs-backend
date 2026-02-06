from typing import List, Dict, Any

from media_rs.rs_types.model import Medium

from api.services.media_data.tmdb import get_multiple_movie_data
from api.services.media_data.book_query import get_multiple_book_data

def get_media_data(
    titles: List[str],
    medium: Medium
) -> Dict[str, Any]:
    if medium == Medium.MOVIES:
        return get_multiple_movie_data(titles)
    elif medium == Medium.BOOKS:
        return get_multiple_book_data(titles)
    else:
        raise ValueError("Invalid medium")