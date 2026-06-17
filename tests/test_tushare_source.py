"""Tests for the Tushare connection helper.

The one behavior pinned here is the proxy bypass that the live debugging session
established: on Windows, popping the proxy *environment* variables is NOT enough,
because `requests` (and therefore tushare's internal `requests.post`) also reads
the system proxy from the registry. A local Clash proxy at 127.0.0.1:7891 then
hijacks the request to the domestic mirror IP and returns 502. The fix is to add
the mirror host to NO_PROXY so the request goes direct.
"""

import os

from ashare_gauntlet.data.tushare_source import make_pro_api


def test_make_pro_api_bypasses_proxy_for_mirror_host_and_pins_url(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7891")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7891")
    monkeypatch.setenv("NO_PROXY", "")  # tracked so monkeypatch restores it

    pro = make_pro_api("tok123", "http://8.148.76.181:8686/")

    # Environment proxies popped.
    assert "HTTP_PROXY" not in os.environ
    assert "https_proxy" not in os.environ
    # Mirror host force-directed (beats the Windows registry proxy too).
    assert "8.148.76.181" in os.environ.get("NO_PROXY", "")
    assert "8.148.76.181" in os.environ.get("no_proxy", "")
    # http_url override applied to the name-mangled attribute.
    assert pro._DataApi__http_url == "http://8.148.76.181:8686/"
