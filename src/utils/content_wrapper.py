from typing import List
from src.types.model import IdType, ContentSimilarity

def get_id_from_similarity_result(
    similarity: List[ContentSimilarity]
) -> List[IdType]:
    id = []
    for result in similarity:
        id += [result[0]]
    return id