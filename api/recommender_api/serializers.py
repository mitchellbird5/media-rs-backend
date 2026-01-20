from rest_framework import serializers

# -----------------------------
# Input serializers
# -----------------------------

class RatingSerializer(serializers.Serializer):
    name = serializers.CharField()
    value = serializers.FloatField(min_value=0, max_value=5)

class UserCFInputSerializer(serializers.Serializer):
    ratings = RatingSerializer(many=True)
    top_n = serializers.IntegerField(default=10, min_value=1)
    k_similar_users = serializers.IntegerField(default=50, min_value=1)
    embedding_method = serializers.ChoiceField(
        choices=["SBERT", "TFIDF"],
        default="SBERT"
    )
    
    
class HybridInputSerializer(serializers.Serializer):
    movie_title = serializers.CharField(required=True)
    alpha = serializers.FloatField(default=0.5, min_value=0, max_value=1)
    beta = serializers.FloatField(default=0.3, min_value=0, max_value=1)
    ratings = RatingSerializer(many=True)
    top_n = serializers.IntegerField(default=10, min_value=1)
    k_similar_users = serializers.IntegerField(default=50, min_value=1)
    embedding_method = serializers.ChoiceField(
        choices=["SBERT", "TFIDF"],
        default="SBERT"
    )
    
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
    embedding_method = serializers.ChoiceField(
        choices=["SBERT", "TFIDF"],
        default="SBERT"
    )
    
class ContentDescriptionInputSerializer(serializers.Serializer):
    description = serializers.CharField()
    top_n = serializers.IntegerField(default=10, min_value=1, max_value=100)
    embedding_method = serializers.ChoiceField(
        choices=["SBERT", "TFIDF"],
        default="SBERT"
    )
    
class MovieSearchInputSerializer(serializers.Serializer):
    query = serializers.CharField()
    limit = serializers.IntegerField(required=False, default=10, min_value=1, max_value=100)

# -----------------------------
# Output serializers
# -----------------------------

class RecommendationSerializer(serializers.Serializer):
    title = serializers.CharField()
    # score = serializers.FloatField()


class RecommendationListSerializer(serializers.Serializer):
    recommendations = RecommendationSerializer(many=True)
