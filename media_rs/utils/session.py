import secrets
from fastapi import Response

SESSION_COOKIE_NAME = "movie_search_sid"

def get_or_create_session_id(response: Response, sid: str | None = None) -> str:
    """
    Return existing session ID or create a new one if missing.
    """
    if not sid:
        sid = secrets.token_urlsafe(32)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=sid,
            httponly=True,
            secure=False,  # Set True in production HTTPS
            samesite="Lax",
            max_age=60 * 60 * 24,  # 24 hours
        )
    return sid
