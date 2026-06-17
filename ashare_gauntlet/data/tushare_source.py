"""Tushare (third-party mirror) connection helper.

The mirror is a domestic-China host that must be reached *directly*. Two things
bite on Windows and are handled here so callers never have to remember them:

1. Popping the proxy environment variables is not sufficient — ``requests`` also
   reads the system proxy from the registry, so a local Clash/VPN proxy still
   hijacks the request. Adding the mirror host to ``NO_PROXY`` forces a direct
   connection regardless.
2. The mirror URL must be assigned to tushare's name-mangled
   ``_DataApi__http_url`` attribute, or the client silently talks to its default
   endpoint.
"""

import os
from urllib.parse import urlparse

import tushare as ts

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def make_pro_api(token: str, http_url: str, timeout: int = 120):
    """Build a tushare ``pro_api`` client pinned to ``http_url`` and reachable
    directly (no proxy) for the mirror host.

    ``timeout`` defaults to 120s rather than tushare's 30s: a full-market
    single-day pull (~5000 rows) can take longer than 30s on a cold call and
    would otherwise raise a ReadTimeout.
    """
    for var in _PROXY_ENV_VARS:
        os.environ.pop(var, None)

    host = urlparse(http_url).hostname
    if host:
        existing = os.environ.get("NO_PROXY", "")
        entries = [e for e in existing.split(",") if e]
        if host not in entries:
            entries.append(host)
        merged = ",".join(entries)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged

    pro = ts.pro_api(token)
    pro._DataApi__http_url = http_url
    pro._DataApi__timeout = timeout
    return pro
