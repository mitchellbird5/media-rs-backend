from rest_framework import serializers

# -----------------------------
# Input serializers
# -----------------------------

class RatingsInputSerializer(serializers.Serializer):
    ratings = serializers.DictField(
        child=serializers.FloatField(),
        help_text="Dictionary of movie title -> rating",
    )
    top_n = serializers.IntegerField(default=10, min_value=1)
    k_similar_users = serializers.IntegerField(default=50, min_value=1)


class TopNInputSerializer(serializers.Serializer):
    movie_title = serializers.CharField(required=False)
    description = serializers.CharField(required=False)
    top_n = serializers.IntegerField(default=10, min_value=1)

# -----------------------------
# Output serializers
# -----------------------------

class RecommendationSerializer(serializers.Serializer):
    title = serializers.CharField()
    # score = serializers.FloatField()


class RecommendationListSerializer(serializers.Serializer):
    recommendations = RecommendationSerializer(many=True)
