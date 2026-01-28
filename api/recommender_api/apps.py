import os
from django.apps import AppConfig

class MediaRSConfig(AppConfig):
    name = "media_rs"

    def ready(self):
        # Only warmup in the child server, skip autoreload parent
        if os.environ.get("RUN_MAIN") != "true":
            return

        from media_rs.utils.movies.movie_data_cache import MovieDataCache
        cache = MovieDataCache(repo_id=os.getenv("HF_REPO_ID"))
        cache.warmup()
        print("MovieDataCache warmup finished")