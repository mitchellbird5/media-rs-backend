from typing import Tuple, Dict

IdType = float | int | str

ContentSimilarity = Tuple[IdType, float]

IdxToTitle = Dict[int, str]
TitleToIdx = Dict[str, int]