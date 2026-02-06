from typing import Dict, Any, Optional

class ItemIndex:
    def __init__(self, data: Dict[str, Any]):
        self.num_items: int = data["num_items"]

        self.idx_to_itemId: Dict[int, int] = data["idx_to_itemId"]
        self.itemId_to_idx: Dict[int, int] = data["itemId_to_idx"]

        self.itemId_to_title: Dict[int, str] = data["itemId_to_title"]
        self.title_to_itemId: Dict[str, int] = data["title_to_itemId"]
        
        self.itemId_to_imdbId: Optional[Dict[int, int]] = data.get("itemId_to_imdbId")
        self.itemId_to_tmdbId: Optional[Dict[int, int]] = data.get("itemId_to_tmdbId")

        self.imdbId_to_itemId: Optional[Dict[int, int]] = data.get("imdbId_to_itemId")
        self.tmdbId_to_itemId: Optional[Dict[int, int]] = data.get("tmdbId_to_itemId")
        
        self.itemId_to_url: Optional[Dict[int, str]] = data.get("itemId_to_url")
        self.url_to_itemId: Optional[Dict[str, int]] = data.get("url_to_itemId")
        
        self.itemId_to_img: Optional[Dict[int, str]] = data.get("itemId_to_img")
        self.img_to_itemId: Optional[Dict[str, int]] = data.get("img_to_itemId")
        
    def title_to_idx(self, title: str) -> int:
        """Convert a movie title to its embedding index"""
        movie_id = self.title_to_itemId.get(title)
        if movie_id is None:
            raise ValueError(f"No movie found with title '{title}'")
        
        idx = self.itemId_to_idx.get(movie_id)
        if idx is None:
            raise ValueError(f"No index found for itemId {movie_id}")
        
        return idx

    def idx_to_title(self, idx: int) -> str:
        """Convert an embedding index to a movie title"""
        movie_id = self.idx_to_itemId.get(idx)
        if movie_id is None:
            raise ValueError(f"No itemId found for index {idx}")
        
        title = self.itemId_to_title.get(movie_id)
        if title is None:
            raise ValueError(f"No title found for itemId {movie_id}")
        
        return title