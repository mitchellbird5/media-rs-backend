from typing import TypedDict, List, Dict
from media_rs.utils.item_index import ItemIndex
class Rating(TypedDict):
    name: str
    value: float
    
def get_index_ratings(
    ratings: List[Rating],
    item_idx: ItemIndex
) -> Dict[int, float]:
    index_ratings = {}

    for r in ratings:
        title = r["name"]
        rating = r["value"]

        # title -> movieIds -> indices
        indices = item_idx.title_to_idx(title)

        if not indices:
            continue

        # If multiple movies share the same title:
        # choose the first one (or apply your own rule)
        idx = indices[0]

        index_ratings[idx] = rating
        
    return index_ratings