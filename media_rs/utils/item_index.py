from typing import Dict, Any, List

from media_rs.utils.movies.load_movie_data import norm

class ItemIndex:
    def __init__(self, data: Dict[str, Any]):
        self.num_items: int = data["num_items"]

        self.idx_to_movieId: Dict[int, int] = data["idx_to_movieId"]
        self.movieId_to_idx: Dict[int, str] = data["movieId_to_idx"]

        self.movieId_to_title: Dict[int, int] = data["movieId_to_title"]
        self.title_to_movieId: Dict[str, int] = data["title_to_movieId"]
        
        self.movieId_to_imdbId: Dict[int, str] = data["movieId_to_imdbId"]
        self.movieId_to_tmdbId: Dict[int, int] = data["movieId_to_tmdbId"]
        
        self.imdbId_to_movieId: Dict[str, int] = data["imdbId_to_movieId"]
        self.tmdbId_to_movieId: Dict[int, int] = data["tmdbId_to_movieId"]

    def title_to_idx(self, title: str) -> List[int]:
        """Return ALL possible item indices for a title"""
        mids = self.title_to_movieId.get(title)
        if not mids:
            raise ValueError(f"No titles found that match '{title}'")
        else: 
            return mids

    def idx_to_title(self, idx: int) -> str:
        return self.movieId_to_title[self.idx_to_movieId[idx]]