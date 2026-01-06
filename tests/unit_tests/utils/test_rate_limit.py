import time
import pytest
from media_rs.utils.rate_limit import RateLimiter

def test_allows_requests_up_to_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=10)
    key = "user-1"

    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is True

def test_blocks_after_limit_exceeded():
    limiter = RateLimiter(max_requests=2, window_seconds=10)
    key = "user-1"

    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False

def test_allows_again_after_window_expires():
    limiter = RateLimiter(max_requests=2, window_seconds=1)
    key = "user-1"

    assert limiter.allow(key) is True
    assert limiter.allow(key) is True
    assert limiter.allow(key) is False

    # Wait for window to expire
    time.sleep(1.1)

    assert limiter.allow(key) is True

def test_separate_keys_have_separate_limits():
    limiter = RateLimiter(max_requests=1, window_seconds=10)

    assert limiter.allow("user-1") is True
    assert limiter.allow("user-1") is False

    assert limiter.allow("user-2") is True
    assert limiter.allow("user-2") is False

def test_sliding_window_behavior():
    limiter = RateLimiter(max_requests=2, window_seconds=1)
    key = "user-1"

    assert limiter.allow(key) is True
    time.sleep(0.6)

    assert limiter.allow(key) is True
    assert limiter.allow(key) is False

    # First request should expire
    time.sleep(0.5)

    assert limiter.allow(key) is True

def test_empty_key_still_rate_limited():
    limiter = RateLimiter(max_requests=1, window_seconds=10)

    assert limiter.allow("") is True
    assert limiter.allow("") is False
