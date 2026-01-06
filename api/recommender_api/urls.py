from django.urls import path
from .views import (
    ContentRecommendationAPI, 
    ContentDescriptionRecommendationAPI,
    ItemCFRecommendationAPI,
    UserCFRecommendationAPI,
    MovieSearchView,
    MovieImagesView
)

urlpatterns = [
    path(
        "recommend/content/", 
        ContentRecommendationAPI.as_view(), 
        name="content-recommend"
    ),
    path(
        "recommend/content-description/", 
        ContentDescriptionRecommendationAPI.as_view(), 
        name="content-recommend-description"
    ),
    path(
        "recommend/item-cf/", 
        ItemCFRecommendationAPI.as_view(), 
        name="item-cf-recommend"
    ),
    path(
        "recommend/user-cf/", 
        UserCFRecommendationAPI.as_view(), 
        name="user-cf-recommend"
    ),
    path(
        "recommend/hybrid/", 
        UserCFRecommendationAPI.as_view(), 
        name="hybrid-recommend"
    ),
    path(
        "movies/search/",
        MovieSearchView.as_view(), 
        name='movie-search'
    ),
    path(
        "movies/images/",
        MovieImagesView.as_view(), 
        name='movie-images'
    )
]
