from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from api.recommender_api.serializers import (
    RatingsInputSerializer,
    TopNInputSerializer,
    RecommendationSerializer,
    RecommendationListSerializer
)
from api.recommender_api.services.content_services import (
    get_content_recommendations,
    get_content_recommendations_from_description
)
from api.recommender_api.services.collab_services import (
    get_item_cf_recommendations,
    get_user_cf_recommendations
)


# -----------------------------
# Content-based
# -----------------------------
class ContentRecommendationAPI(APIView):
    def get(self, request):
        serializer = TopNInputSerializer(data=request.query_params)
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
        serializer = TopNInputSerializer(data=request.query_params)
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
        serializer = TopNInputSerializer(data=request.query_params)
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
        serializer = RatingsInputSerializer(data=request.data)
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
