"""Proxy URL discovery from standard env vars.

aiohttp respects HTTP_PROXY / HTTPS_PROXY only when `trust_env=True` is set on
the ClientSession. `websockets` ignores env vars entirely — callers must pass
`proxy=` explicitly. This helper centralises the env lookup so both code paths
agree on which proxy to use.
"""

from __future__ import annotations

import os


def get_proxy_url() -> str | None:
    """Return the HTTPS proxy URL from env (if any), else None.

    Order mirrors `curl`: HTTPS_PROXY / https_proxy first, then ALL_PROXY /
    all_proxy, then HTTP_PROXY / http_proxy as a fallback.
    """
    for key in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(key)
        if v:
            return v
    return None
