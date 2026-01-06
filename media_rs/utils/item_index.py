import pickle
from typing import Dict, Any

class ItemIndex:
    def __init__(self, data: Dict[str, Any]):
        self.num_items: int = data["num_items"]

        self.idx_to_movieId: Dict[int, int] = data["idx_to_movieId"]
        self.idx_to_title: Dict[int, str] = data["idx_to_title"]

        self.movieId_to_idx: Dict[int, int] = data["movieId_to_idx"]
        self.title_to_idx: Dict[str, int] = data["title_to_idx"]
        
        self.movieId_to_imdbId: Dict[int, str] = data["movieId_to_imdbId"]
        self.movieId_to_tmdbId: Dict[int, int] = data["movieId_to_tmdbId"]
        
        self.imdbId_to_movieId: Dict[str, int] = data["imdbId_to_movieId"]
        self.tmdbId_to_movieId: Dict[int, int] = data["tmdbId_to_movieId"]

    @property
    def ids(self):
        return list(range(self.num_items))