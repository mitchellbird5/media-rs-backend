# tests/test_api_e2e.py
import pytest
from rest_framework.test import APIClient

@pytest.mark.django_db
def test_content_api_e2e(api_client: APIClient):
    url = "/api/recommend/content/"
    response = api_client.get(url, {"movie_title": "Toy Story (1995)", "top_n": 2})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data)!=0


@pytest.mark.django_db
def test_content_description_api_e2e(api_client: APIClient):
    url = "/api/recommend/content-description/"
    response = api_client.get(url, {"description": "fun movie", "top_n": 2})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data)!=0


@pytest.mark.django_db
def test_item_cf_api_e2e(api_client: APIClient):
    url = "/api/recommend/item-cf/"
    response = api_client.get(url, {"movie_title": "Toy Story (1995)", "top_n": 2})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data)!=0


@pytest.mark.django_db
def test_user_cf_api_e2e(api_client: APIClient):
    url = "/api/recommend/user-cf/"
    payload = {
        "ratings": {'Toy Story 2 (1999)': 5, 'Forrest Gump (1994)': 3},
        "top_n": 2,
        "k_similar_users": 2
    }
    response = api_client.post(url, payload, format="json")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data)!=0


@pytest.mark.django_db
def test_hybrid_api_e2e(api_client: APIClient):
    url = "/api/recommend/hybrid/"
    payload = {
        "movie_title": "Toy Story (1995)",
        "ratings": {'Toy Story 2 (1999)': 5},
        "alpha": 0.4,
        "beta": 0.3,
        "top_n": 2,
        "k_similar_users": 2
    }
    response = api_client.post(url, payload, format="json")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data)!=0