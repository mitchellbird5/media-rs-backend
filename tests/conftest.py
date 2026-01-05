# tests/conftest.py
import pytest
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache

@pytest.fixture(scope="session", autouse=True)
def preload_movie_cache():
    """Download and cache movie data once per CI session."""
    print("Downloading movie data cache for tests...")
    get_movie_data_cache()
    print("Movie data cache ready")
