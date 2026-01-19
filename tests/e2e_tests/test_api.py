import pytest
from rest_framework.test import APIClient
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache

@pytest.fixture
def api_client():
    return APIClient()

@pytest.mark.django_db
def test_content_api_e2e(api_client: APIClient):
    # Load movie cache only when needed
    movie_cache = get_movie_data_cache()
    
    url = "/api/recommend/content/"
    response = api_client.get(url, {"movie_title": "Toy Story (1995)", "top_n": 2})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


@pytest.mark.django_db
def test_content_description_api_e2e(api_client: APIClient):
    movie_cache = get_movie_data_cache()
    
    url = "/api/recommend/content-description/"
    response = api_client.get(url, {"description": "fun movie", "top_n": 2})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


@pytest.mark.django_db
def test_item_cf_api_e2e(api_client: APIClient):
    movie_cache = get_movie_data_cache()
    
    url = "/api/recommend/item-cf/"
    response = api_client.get(url, {"movie_title": "Toy Story (1995)", "top_n": 2})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


@pytest.mark.django_db
def test_user_cf_api_e2e(api_client: APIClient): 
    url = "/api/recommend/user-cf/"
    payload = {
        "ratings": [
            {
                'name': 'Toy Story 2 (1999)',
                'value': 5, 
            },
            {
                'name':'Forrest Gump (1994)',
                'value': 3
            }
        ],
        "top_n": 2,
        "k_similar_users": 2
    }
    response = api_client.post(url, payload, format="json")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


@pytest.mark.django_db
def test_hybrid_api_e2e(api_client: APIClient):
    url = "/api/recommend/hybrid/"
    payload = {
        "movie_title": "Toy Story (1995)",
        "ratings": [
            {
                'name': 'Toy Story 2 (1999)',
                'value': 5, 
            },
            {
                'name':'Forrest Gump (1994)',
                'value': 3
            }
        ],
        "alpha": 0.4,
        "beta": 0.3,
        "top_n": 2,
        "k_similar_users": 2
    }
    response = api_client.post(url, payload, format="json")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) != 0


@pytest.mark.django_db
def test_movie_search_api_e2e(api_client: APIClient):
    # Only load cache if needed (not strictly needed here if API doesn't use it)
    # movie_cache = get_movie_data_cache()
    
    url = "/api/movies/search/"
    response = api_client.get(url, {"query": "Toy Story", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 5, "Expected at least one movie in the response"

    titles = [movie["title"] for movie in data]
    assert any("Toy Story" in title for title in titles), "Expected 'Toy Story' in results"

@pytest.mark.django_db
def test_movie_data_api_e2e(api_client: APIClient):
    url = "/api/movies/data/"
    response = api_client.get(url, {"titles": ["Toy Story (1995)", "Forrest Gump (1994)"]})
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) == 2

    titles_returned = [item.get("title") for item in data]
    assert "Toy Story (1995)" in titles_returned
    assert "Forrest Gump (1994)" in titles_returned

    for item in data:
        assert "title" in item
        # These can be None
        assert "poster_path" in item
        assert "backdrop_path" in item
        # Optional type-safe checks
        assert isinstance(item.get("genres", {}), dict)