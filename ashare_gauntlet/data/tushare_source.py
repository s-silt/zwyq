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


def make_pro_api(token: str, http_url: str | None = None, timeout: int = 120):
    """Build a Tushare client, optionally pinned directly to a mirror.

    With no explicit URL, the SDK's official endpoint and the caller's proxy
    environment are left untouched.  Mirror mode retains the Windows direct-
    connection workaround and the generous timeout needed by full-market pulls.
    """
    mirror_url = http_url.strip() if http_url else ""
    if not mirror_url:
        return ts.pro_api(token)

    for var in _PROXY_ENV_VARS:
        os.environ.pop(var, None)

    host = urlparse(mirror_url).hostname
    if host:
        existing = os.environ.get("NO_PROXY", "")
        entries = [e for e in existing.split(",") if e]
        if host not in entries:
            entries.append(host)
        merged = ",".join(entries)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged

    pro = ts.pro_api(token)
    pro._DataApi__http_url = mirror_url
    pro._DataApi__timeout = timeout
    return pro
