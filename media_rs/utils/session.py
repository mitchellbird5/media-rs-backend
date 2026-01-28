import secrets

SESSION_COOKIE_NAME = "movie_search_sid"

def get_or_create_session_id(request, response):
    sid = request.COOKIES.get(SESSION_COOKIE_NAME)

    if not sid:
        sid = secrets.token_urlsafe(32)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=sid,
            httponly=True,
            secure=True,      # set False only for local HTTP
            samesite="Lax",
            max_age=60 * 60 * 24  # 24 hours
        )

    return sid
