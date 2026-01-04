import os
from supabase import create_client, Client
from typing import List, Dict, Any

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class MoviesService:
    """Service responsible for querying movies from Supabase."""

    @staticmethod
    def search_movies(query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search for movies by title (case-insensitive)."""
        if not query or len(query.strip()) < 2:
            return []

        try: 
            response = supabase.table(SUPABASE_TABLE) \
                .select("movieId, title") \
                .ilike("title", f"%{query}%") \
                .limit(limit) \
                .execute()
            return response.data
        
        except:
            raise Exception(f"Supabase error")

        