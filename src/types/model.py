from typing import Tuple

from enum import Enum

IdType = float | int | str

ContentSimilarity = Tuple[IdType, float]

class CollabMethod(str, Enum):
    ITEM = "ITEM",
    USER = "USER"