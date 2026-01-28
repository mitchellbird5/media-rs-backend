import os
from supabase import create_client, Client
from typing import List, Dict, Any

from media_rs.utils.rate_limit import RateLimiter

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE")

rate_limiter = RateLimiter(
    max_requests=20, 
    window_seconds=1    
)

try:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase ping successful")
except Exception as e:
    print("Supabase ping failed:", e)


class MoviesService:
    """Service responsible for querying movies from Supabase."""

    @staticmethod
    def search_movies(
        query: str,
        user_key: str,   # IP address or user ID
        limit: int
    ) -> List[Dict[str, Any]]:

        if not query or len(query.strip()) < 2:
            return []

        if not rate_limiter.allow(user_key):
            # Fail fast – don't hit Supabase
            raise Exception("Rate limit exceeded")

        try:
            response = (
                client
                .table(SUPABASE_TABLE)
                .select("movieId, title")
                .ilike("title", f"%{query}%")
                .order("title")
                .limit(limit)
                .execute()
            )
            return response.data[:limit]

        except Exception as e:
            raise Exception(f"Supabase error: {e}")