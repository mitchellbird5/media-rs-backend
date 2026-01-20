from typing import Tuple, Dict

from enum import Enum

IdType = float | int | str

ContentSimilarity = Tuple[IdType, float]

class EmbeddingMethod(Enum):
    SBERT = 1
    TFIDF = 2

IdxToTitle = Dict[int, str]
TitleToIdx = Dict[str, int]