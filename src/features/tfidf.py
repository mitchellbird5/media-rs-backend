import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

def compute_tfidf_matrix(combined_text: List[str], max_features=5000) -> np.ndarray:
    """
    Returns the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    return tfidf_matrix