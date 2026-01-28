from typing import Dict, Any, List

from media_rs.utils.movies.load_movie_data import norm

class ItemIndex:
    def __init__(self, data: Dict[str, Any]):
        self.num_items: int = data["num_items"]

        self.idx_to_movieId: Dict[int, int] = data["idx_to_movieId"]
        self.movieId_to_idx: Dict[int, int] = data["movieId_to_idx"]

        self.movieId_to_title: Dict[int, str] = data["movieId_to_title"]
        self.title_to_movieId: Dict[str, int] = data["title_to_movieId"]
        
        self.movieId_to_imdbId: Dict[int, int] = data["movieId_to_imdbId"]
        self.movieId_to_tmdbId: Dict[int, int] = data["movieId_to_tmdbId"]
        
        self.imdbId_to_movieId: Dict[int, int] = data["imdbId_to_movieId"]
        self.tmdbId_to_movieId: Dict[int, int] = data["tmdbId_to_movieId"]
        
    def title_to_idx(self, title: str) -> int:
        """Convert a movie title to its embedding index"""
        movie_id = self.title_to_movieId.get(title)
        if movie_id is None:
            raise ValueError(f"No movie found with title '{title}'")
        
        idx = self.movieId_to_idx.get(movie_id)
        if idx is None:
            raise ValueError(f"No index found for movieId {movie_id}")
        
        return idx

    def idx_to_title(self, idx: int) -> str:
        """Convert an embedding index to a movie title"""
        movie_id = self.idx_to_movieId.get(idx)
        if movie_id is None:
            raise ValueError(f"No movieId found for index {idx}")
        
        title = self.movieId_to_title.get(movie_id)
        if title is None:
            raise ValueError(f"No title found for movieId {movie_id}")
        
        return title