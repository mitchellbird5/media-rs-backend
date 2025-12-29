import pickle
from pathlib import Path
from typing import Dict

class ItemIndex:
    def __init__(self, path: Path):
        data = pickle.load(open(path, "rb"))

        self.num_items: int = data["num_items"]

        self.idx_to_movieId: Dict[int, int] = data["idx_to_movieId"]
        self.idx_to_title: Dict[int, str] = data["idx_to_title"]

        self.movieId_to_idx: Dict[int, int] = data["movieId_to_idx"]
        self.title_to_idx: Dict[str, int] = data["title_to_idx"]

    @property
    def ids(self):
        return list(range(self.num_items))