#!/usr/bin/env bash
set -e

echo "▶ Starting container entrypoint"

# -----------------------------
# Warm movie data cache
# -----------------------------
echo "▶ Warming movie data cache..."
python - <<'EOF'
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache

cache = get_movie_data_cache()
print("✔ Movie data cache loaded")
print(f"✔ Loaded files: {list(cache.data.keys())}")
EOF

echo "▶ Cache warm complete"

# -----------------------------
# Start Django
# -----------------------------
echo "▶ Starting Django server"
exec "$@"
