import time
import threading
from collections import defaultdict, deque

class RateLimiter:
    """
    Allows `max_requests` per `window_seconds` per key.
    """
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            q = self.requests[key]

            # Remove expired timestamps
            while q and q[0] <= now - self.window_seconds:
                q.popleft()

            if len(q) >= self.max_requests:
                return False

            q.append(now)
            return True
