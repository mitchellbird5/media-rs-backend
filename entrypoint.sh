#!/bin/bash
set -e

echo "Preloading Hugging Face cache..."
poetry run python media_rs/utils/movies/movie_data_cache.py
