import os
from supabase import create_client, Client
from typing import List, Dict, Any

from media_rs.utils.rate_limit import RateLimiter
from media_rs.rs_types.model import Medium

from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 
SUPABASE_MOVIE_TABLE = os.getenv("SUPABASE_MOVIE_TABLE")
SUPABASE_BOOK_TABLE = os.getenv("SUPABASE_BOOK_TABLE")

rate_limiter = RateLimiter(
    max_requests=20, 
    window_seconds=1    
)

try:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase ping successful")
except Exception as e:
    print("Supabase ping failed:", e)


class DatabaseService:
    """Service responsible for querying movies and books from Supabase."""

    @staticmethod
    def search_database(
        query: str,
        user_key: str,   # IP address or user ID
        medium: Medium,
        limit: int
    ) -> List[Dict[str, Any]]:

        if not query or len(query.strip()) < 2:
            return []

        if not rate_limiter.allow(user_key):
            # Fail fast – don't hit Supabase
            raise Exception("Rate limit exceeded")

        print("medium =", medium)
        print("SUPABASE_MOVIE_TABLE =", SUPABASE_MOVIE_TABLE)
        print("SUPABASE_BOOK_TABLE =", SUPABASE_BOOK_TABLE)

        if medium == Medium.MOVIES:
            print('Setting table to ', SUPABASE_MOVIE_TABLE)
            table = SUPABASE_MOVIE_TABLE
        elif medium == Medium.BOOKS:
            print('Setting table to ', SUPABASE_BOOK_TABLE)
            table = SUPABASE_BOOK_TABLE
        else:
            print('Invalid medium:', medium)
            raise ValueError("Invalid medium")

        print('table =', table)

        try:
            response = (
                client
                .table(table)
                .select("itemId, title")
                .ilike("title", f"%{query}%")
                .order("title")
                .limit(limit)
                .execute()
            )
            return response.data[:limit]

        except Exception as e:
            raise Exception(f"Supabase error: {e}")