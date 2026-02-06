import pandas as pd

def build_book_item_index(books: pd.DataFrame):
  
    idx_to_itemId = dict(enumerate(books["itemId"].values))
    itemId_to_idx = {mid: idx for idx, mid in idx_to_itemId.items()}

    itemId_to_title = dict(zip(books["itemId"], books["title"]))
    title_to_itemId = {title: mid for mid, title in itemId_to_title.items()}

    return {
        "num_items": len(books),

        # index layer
        "idx_to_itemId": idx_to_itemId,
        "itemId_to_idx": itemId_to_idx,

        # presentation layer
        "itemId_to_title": itemId_to_title,
        "title_to_itemId": title_to_itemId,
    }
