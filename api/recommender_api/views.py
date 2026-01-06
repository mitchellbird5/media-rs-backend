from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse

from api.recommender_api.serializers import (
    ContentRecommendationInputSerializer,
    ContentDescriptionInputSerializer,
    UserCFInputSerializer,
    HybridInputSerializer,
)
from api.recommender_api.services.content_services import (
    get_content_recommendations,
    get_content_recommendations_from_description,
)
from api.recommender_api.services.collab_services import (
    get_item_cf_recommendations,
    get_user_cf_recommendations
)
from api.recommender_api.services.hybrid_services import (
    get_hybrid_recommendations
)
from api.recommender_api.services.tmdb import (
    get_multiple_movie_images
)
from api.recommender_api.services.database_query import MoviesService


from media_rs.utils.session import get_or_create_session_id


# -----------------------------
# Content-based
# -----------------------------
class ContentRecommendationAPI(APIView):
    def get(self, request):
        serializer = ContentRecommendationInputSerializer(data=request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        movie_title = data.get("movie_title")
        top_n = data["top_n"]

        if not movie_title:
            return Response({"error": "movie_title parameter required"}, status=status.HTTP_400_BAD_REQUEST)

        recs = get_content_recommendations(movie_title, top_n)
        return Response(recs)


# -----------------------------
# Content-from-description
# -----------------------------
class ContentDescriptionRecommendationAPI(APIView):
    def get(self, request):
        serializer = ContentDescriptionInputSerializer(data=request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        description = data.get("description")
        top_n = data["top_n"]

        if not description:
            return Response({"error": "description parameter required"}, status=status.HTTP_400_BAD_REQUEST)

        recs = get_content_recommendations_from_description(description, top_n)

        return Response(recs)

# -----------------------------
# Item CF
# -----------------------------
class ItemCFRecommendationAPI(APIView):
    def get(self, request):
        serializer = ContentRecommendationInputSerializer(data=request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        movie_title = data.get("movie_title")
        top_n = data["top_n"]

        if not movie_title:
            return Response({"error": "movie_title parameter required"}, status=status.HTTP_400_BAD_REQUEST)

        recs = get_item_cf_recommendations(movie_title, top_n)

        return Response(recs)


# -----------------------------
# User CF
# -----------------------------
class UserCFRecommendationAPI(APIView):
    def post(self, request):
        serializer = UserCFInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        ratings = data["ratings"]
        top_n = data["top_n"]
        k_similar_users = data["k_similar_users"]

        try:
            recs = get_user_cf_recommendations(
                ratings=ratings,
                top_n=top_n,
                k_similar_users=k_similar_users
            )
        except Exception as e:
            return Response(
                {"error": f"Failed to get recommendations: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(recs)

class HybridRecommendationAPI(APIView):
    def post(self, request):
        serializer = HybridInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        movie_title = data["movie_title"]
        alpha = data["alpha"]
        beta = data["beta"]
        ratings = data["ratings"]
        top_n = data["top_n"]
        k_similar_users = data["k_similar_users"]

        try:
            recs = get_hybrid_recommendations(
                movie_title=movie_title,
                ratings=ratings,
                alpha=alpha,
                beta=beta,
                top_n=top_n,
                k_similar_users=k_similar_users
            )
        except Exception as e:
            return Response(
                {"error": f"Failed to get recommendations: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(recs)
    
class MovieSearchView(APIView):
    def get(self, request):
        query = request.GET.get("query", "").strip()

        response = JsonResponse({}, safe=False)

        try:
            session_id = get_or_create_session_id(request, response)

            results = MoviesService.search_movies(
                query=query,
                user_key=f"session:{session_id}"
            )

            response.content = JsonResponse(results, safe=False).content
            return response

        except Exception as e:
            response.status_code = 429 if "Rate limit" in str(e) else 500
            response.content = JsonResponse(
                {"error": str(e)}
            ).content
            return response
        
class MovieImagesView(APIView):
    def get(self, request):
        titles = request.GET.getlist("titles")

        if not titles:
            return Response(
                {"error": "At least one title parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            images = get_multiple_movie_images(titles)
            return Response(images)

        except Exception as e:
            return Response(
                {"error": f"Failed to get movie images: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )