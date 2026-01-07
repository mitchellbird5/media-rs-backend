from rest_framework import serializers

# -----------------------------
# Input serializers
# -----------------------------

class UserCFInputSerializer(serializers.Serializer):
    ratings = serializers.DictField(
        child=serializers.FloatField(),
        help_text="Dictionary of movie title -> rating",
    )
    top_n = serializers.IntegerField(default=10, min_value=1)
    k_similar_users = serializers.IntegerField(default=50, min_value=1)
    
    
class HybridInputSerializer(serializers.Serializer):
    movie_title = serializers.CharField(required=True)
    alpha = serializers.FloatField(default=0.5, min_value=0, max_value=1)
    beta = serializers.FloatField(default=0.3, min_value=0, max_value=1)
    ratings = serializers.DictField(
        child=serializers.FloatField(),
        help_text="Dictionary of movie title -> rating",
    )
    top_n = serializers.IntegerField(default=10, min_value=1)
    k_similar_users = serializers.IntegerField(default=50, min_value=1)
    
    def validate(self, attrs):
        alpha = attrs.get("alpha", 0)
        beta = attrs.get("beta", 0)

        if alpha + beta > 1:
            raise serializers.ValidationError(
                "alpha + beta must be less than or equal to 1"
            )

        return attrs


class ContentRecommendationInputSerializer(serializers.Serializer):
    movie_title = serializers.CharField()
    top_n = serializers.IntegerField(default=10, min_value=1, max_value=100)
    
class ContentDescriptionInputSerializer(serializers.Serializer):
    description = serializers.CharField()
    top_n = serializers.IntegerField(default=10, min_value=1, max_value=100)
    
class MovieSearchInputSerializer(serializers.Serializer):
    query = serializers.CharField()
    limit = serializers.IntegerField(default=10, min_value=1, max_value=100)

# -----------------------------
# Output serializers
# -----------------------------

class RecommendationSerializer(serializers.Serializer):
    title = serializers.CharField()
    # score = serializers.FloatField()


class RecommendationListSerializer(serializers.Serializer):
    recommendations = RecommendationSerializer(many=True)
