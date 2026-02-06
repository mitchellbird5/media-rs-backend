import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.fixture(scope="session")
def api_client():
    return TestClient(app)


# -----------------------------
# Shared test data
# -----------------------------
CONTENT_TITLES = {
    "movies": "Toy Story (1995)",
    "books": "Don Quixote",
}

DESCRIPTIONS = {
    "movies": "fun animated movie",
    "books": "fantasy novel about a young wizard",
}

USER_RATINGS = {
    "movies": [
        {"name": "Toy Story 2 (1999)", "value": 5},
        {"name": "Forrest Gump (1994)", "value": 3},
    ],
    "books": [
        {"name": "Don Quixote", "value": 5},
        {"name": "The Count of Monte Cristo", "value": 4},
    ],
}


# -----------------------------
# Content-based recommendation
# -----------------------------
@pytest.mark.parametrize("embedding_method", ["SBERT", "TFIDF"])
@pytest.mark.parametrize("medium", ["movies", "books"])
def test_content_api_e2e(api_client, embedding_method, medium):
    response = api_client.get(
        "/api/recommend/content",
        params={
            "title": CONTENT_TITLES[medium],
            "top_n": 2,
            "embedding_method": embedding_method,
            "medium": medium,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


# -----------------------------
# Content-based recommendation from description
# -----------------------------
@pytest.mark.parametrize("embedding_method", ["SBERT", "TFIDF"])
@pytest.mark.parametrize("medium", ["movies", "books"])
def test_content_description_api_e2e(api_client, embedding_method, medium):
    response = api_client.get(
        "/api/recommend/content-description",
        params={
            "description": DESCRIPTIONS[medium],
            "top_n": 2,
            "embedding_method": embedding_method,
            "medium": medium,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


# -----------------------------
# Item-based collaborative filtering
# -----------------------------
@pytest.mark.parametrize("medium", ["movies", "books"])
def test_item_cf_api_e2e(api_client, medium):
    response = api_client.get(
        "/api/recommend/item-cf",
        params={
            "title": CONTENT_TITLES[medium],
            "top_n": 2,
            "medium": medium,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


# -----------------------------
# User-based collaborative filtering
# -----------------------------
@pytest.mark.parametrize("embedding_method", ["SBERT", "TFIDF"])
@pytest.mark.parametrize("medium", ["movies", "books"])
def test_user_cf_api_e2e(api_client, embedding_method, medium):
    payload = {
        "ratings": USER_RATINGS[medium],
        "top_n": 2,
        "k_similar_users": 2,
        "embedding_method": embedding_method,
        "medium": medium,
    }

    response = api_client.post(
        "/api/recommend/user-cf",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


# -----------------------------
# Hybrid recommendation
# -----------------------------
@pytest.mark.parametrize("embedding_method", ["SBERT", "TFIDF"])
@pytest.mark.parametrize("medium", ["movies", "books"])
def test_hybrid_api_e2e(api_client, embedding_method, medium):
    payload = {
        "title": CONTENT_TITLES[medium],
        "ratings": USER_RATINGS[medium],
        "alpha": 0.4,
        "beta": 0.3,
        "top_n": 2,
        "k_similar_users": 2,
        "embedding_method": embedding_method,
        "medium": medium,
    }

    response = api_client.post(
        "/api/recommend/hybrid",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


# -----------------------------
# Medium search
# -----------------------------
@pytest.mark.parametrize("medium", ["movies", "books"])
def test_medium_search_api_e2e(api_client, medium):
    response = api_client.get(
        "/api/medium/search",
        params={
            "query": "Toy Story",
            "limit": 5,
            "media": "movies",
            "medium": medium,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 5
    assert any("Toy Story" in item["title"] for item in data)


# -----------------------------
# Movie data 
# -----------------------------
def test_movie_data_api_e2e(api_client):
    response = api_client.get(
        "/api/movies/data",
        params={
            "titles": [
                "Toy Story (1995)",
                "Forrest Gump (1994)",
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2

    for item in data:
        assert item.get("title") is not None
        assert item.get("poster_path") is not None
        assert item.get("backdrop_path") is not None
        assert item.get("genres") is not None
        assert isinstance(item["genres"], dict)
