from django.apps import AppConfig


class MediaRSConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api.recommender_api"
    label = "recommender_api_app"

    def ready(self):
        from media_rs.utils.movies.movie_data_cache import get_movie_data_cache
        get_movie_data_cache()