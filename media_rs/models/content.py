# src/models/content.py
import numpy as np

from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from media_rs.rs_types.model import IdType, ContentSimilarity


class ContentModel:
    def __init__(
        self,
        ids: List[IdType],
        topk_graph:  Dict[int, List[Tuple[int, float]]],
        embeddings: np.ndarray,
        vectorizer: TfidfVectorizer,
        svd: TruncatedSVD
    ):
        self.ids = ids
        self.topk_graph = topk_graph
        self.embeddings = embeddings
        self.vectorizer = vectorizer
        self.svd = svd

    def recommend(self, item_id: IdType, top_n: int) -> List[ContentSimilarity]:
        return self.topk_graph[item_id][:top_n]

    def recommend_from_text(self, text: str, top_n: int = 10) -> List[ContentSimilarity]:
        if self.vectorizer is None or self.svd is None or self.embeddings is None:
            raise RuntimeError("Cold-start not enabled")

        vec = self.vectorizer.transform([text])
        emb = self.svd.transform(vec).astype("float32")

        sims = emb @ self.embeddings.T
        top_idx = np.argsort(-sims[0])[:top_n]

        return [(self.ids[i], float(sims[0][i])) for i in top_idx]
