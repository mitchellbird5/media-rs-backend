import pytest
from fastapi.testclient import TestClient
from api.app import app

@pytest.fixture(scope="session")
def api_client():
    return TestClient(app)

@pytest.mark.parametrize("embedding_method", ["SBERT", "TFIDF"])
def test_content_api_e2e(api_client, embedding_method):
    response = api_client.get(
        "/api/recommend/content",
        params={
            "movie_title": "Toy Story (1995)",
            "top_n": 2,
            "embedding_method": embedding_method,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) != 0



@pytest.mark.parametrize("embedding_method", ["SBERT", "TFIDF"])
def test_content_description_api_e2e(api_client, embedding_method):
    response = api_client.get(
        "/api/recommend/content-description",
        params={
            "description": "fun animated movie",
            "top_n": 2,
            "embedding_method": embedding_method,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) != 0




def test_item_cf_api_e2e(api_client):
    response = api_client.get(
        "/api/recommend/item-cf",
        params={
            "movie_title": "Toy Story (1995)",
            "top_n": 2,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) != 0



@pytest.mark.parametrize("embedding_method", ["SBERT", "TFIDF"])
def test_user_cf_api_e2e(api_client, embedding_method):
    payload = {
        "ratings": [
            {"name": "Toy Story 2 (1999)", "value": 5},
            {"name": "Forrest Gump (1994)", "value": 3},
        ],
        "top_n": 2,
        "k_similar_users": 2,
        "embedding_method": embedding_method,
    }

    response = api_client.post(
        "/api/recommend/user-cf",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) != 0




@pytest.mark.parametrize("embedding_method", ["SBERT", "TFIDF"])
def test_hybrid_api_e2e(api_client, embedding_method):
    payload = {
        "movie_title": "Toy Story (1995)",
        "ratings": [
            {"name": "Toy Story 2 (1999)", "value": 5},
            {"name": "Forrest Gump (1994)", "value": 3},
        ],
        "alpha": 0.4,
        "beta": 0.3,
        "top_n": 2,
        "k_similar_users": 2,
        "embedding_method": embedding_method,
    }

    response = api_client.post(
        "/api/recommend/hybrid",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) != 0


def test_movie_search_api_e2e(api_client):
    response = api_client.get(
        "/api/movies/search",
        params={
            "query": "Toy Story",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 5

    titles = [movie["title"] for movie in data]
    assert any("Toy Story" in title for title in titles)


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
