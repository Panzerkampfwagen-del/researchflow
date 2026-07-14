"""Shared slowapi limiter used to throttle expensive endpoints.

Lives in its own module so both ``app.main`` (which wires it onto the app and
registers the ``RateLimitExceeded`` handler) and the routers (which decorate
individual endpoints) can import it without a circular dependency.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# Keyed by client IP. Endpoints opt in via ``@limiter.limit(...)``; anything
# left undecorated stays unlimited.
limiter = Limiter(key_func=get_remote_address)
