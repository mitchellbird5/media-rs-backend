import numpy as np
import pytest

from features.tfidf import (
    compute_tfidf_matrix
)


def test_compute_tfidf_matrix_basic():
    texts = [
        "apple banana",
        "banana orange",
        "apple orange banana",
    ]

    matrix = compute_tfidf_matrix(texts)

    # Should return a sparse matrix with one row per document
    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] > 0

def test_compute_tfidf_matrix_identical_texts():
    texts = [
        "same text",
        "same text",
    ]

    matrix = compute_tfidf_matrix(texts)

    # Rows should be identical
    np.testing.assert_allclose(
        matrix.toarray()[0],
        matrix.toarray()[1],
    )
    
def test_compute_tfidf_matrix_empty_input():
    with pytest.raises(ValueError):
        compute_tfidf_matrix([])

def test_compute_tfidf_matrix_max_features():
    texts = [
        "apple banana orange pear mango",
        "banana mango kiwi pineapple",
    ]

    matrix = compute_tfidf_matrix(texts, max_features=2)

    assert matrix.shape[1] <= 2