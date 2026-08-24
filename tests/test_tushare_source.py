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


def test_make_pro_api_without_mirror_preserves_sdk_defaults_and_proxy(monkeypatch):
    class FakePro:
        pass

    fake = FakePro()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7891")
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setattr("ashare_gauntlet.data.tushare_source.ts.pro_api", lambda token: fake)

    assert make_pro_api("tok123", None) is fake
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7891"
    assert os.environ["NO_PROXY"] == "localhost"
    assert not hasattr(fake, "_DataApi__http_url")
    assert not hasattr(fake, "_DataApi__timeout")


def test_make_pro_api_blank_mirror_uses_official_mode(monkeypatch):
    class FakePro:
        pass

    fake = FakePro()
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy")
    monkeypatch.setattr("ashare_gauntlet.data.tushare_source.ts.pro_api", lambda token: fake)
    assert make_pro_api("tok123", "   ") is fake
    assert os.environ["HTTPS_PROXY"] == "http://proxy"


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


def test_make_pro_api_sets_generous_timeout_for_full_market_pulls(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "")
    # The default 30s timeout is too short for a full-market single-day pull
    # (~5000 rows) on a cold call — it must default to something generous.
    pro_default = make_pro_api("tok", "http://1.2.3.4:8080")
    assert pro_default._DataApi__timeout >= 120

    pro_explicit = make_pro_api("tok", "http://1.2.3.4:8080", timeout=200)
    assert pro_explicit._DataApi__timeout == 200


# ── fail-loud:SDK 把 HTTP 错误吞成空表(实测 403 静默返回空 DataFrame)──
#
# tushare 的 DataApi.query 写的是 `res = requests.post(...); if res: ... else:
# return pd.DataFrame()`,而 Response.__bool__ 即 res.ok —— 403/500 全部落进
# else,调用方看到的是"今天没有数据"。判据用 SDK 自身的两条路径区分:
# 成功=pd.DataFrame(items, columns=fields) 必带列名;失败=裸 pd.DataFrame() 零列。

import pandas as pd
import pytest

from ashare_gauntlet.data.tushare_source import (
    TushareTransportError,
    _install_fail_loud_query,
)


class _FakeDataApi:
    """最小 DataApi 替身:复刻 __getattr__ → partial(self.query, name) 的分发。"""

    def __init__(self, result):
        self._result = result
        self.calls = []
        self._DataApi__http_url = "https://mirror.example.com"
        self._DataApi__token = "SECRET-TOKEN-abc123"

    def query(self, api_name, fields="", **kwargs):
        self.calls.append((api_name, fields, kwargs))
        return self._result

    def __getattr__(self, name):
        from functools import partial
        return partial(self.query, name)


def test_swallowed_http_error_raises_instead_of_empty_frame(monkeypatch):
    """★零列空表=传输失败,必须抛错;绝不能当成'今日无数据'返回给调用方。"""
    monkeypatch.setattr(
        "ashare_gauntlet.data.tushare_source._probe_status", lambda url, timeout=10: "HTTP 403")
    pro = _install_fail_loud_query(_FakeDataApi(pd.DataFrame()))

    with pytest.raises(TushareTransportError) as exc:
        pro.stock_basic(list_status="L")

    msg = str(exc.value)
    assert "stock_basic" in msg          # 哪个接口
    assert "HTTP 403" in msg             # 可行动的状态码
    assert "mirror.example.com" in msg   # 哪个端点
    # 绝不泄漏 token 明文(消息里出现"token 有效期"这类提示文案是允许的)
    assert "SECRET-TOKEN-abc123" not in msg


def test_genuine_empty_result_with_columns_passes_through():
    """真实的空结果(带列名)不能被误判为故障——非交易日/无记录是合法返回。"""
    empty_but_typed = pd.DataFrame(columns=["ts_code", "name", "industry"])
    pro = _install_fail_loud_query(_FakeDataApi(empty_but_typed))

    out = pro.stock_basic(list_status="L")
    assert list(out.columns) == ["ts_code", "name", "industry"]
    assert len(out) == 0


def test_normal_result_passes_through_untouched():
    df = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [12.3]})
    pro = _install_fail_loud_query(_FakeDataApi(df))
    out = pro.daily(trade_date="20260821")
    assert out.equals(df)


def test_wrapper_covers_every_endpoint_via_getattr_dispatch():
    """实例属性优先于类属性 → 所有接口名都走包装后的 query,不只 stock_basic。"""
    fake = _FakeDataApi(pd.DataFrame({"a": [1]}))
    pro = _install_fail_loud_query(fake)
    pro.daily(trade_date="20260821")
    pro.trade_cal(start_date="20260801")
    pro.adj_factor(trade_date="20260821")
    assert [c[0] for c in fake.calls] == ["daily", "trade_cal", "adj_factor"]


def test_client_without_query_returned_untouched():
    """测试替身/未来 SDK 变体没有 query 时原样返回,不制造新的失败模式。"""
    class Bare:
        pass

    bare = Bare()
    assert _install_fail_loud_query(bare) is bare


def test_probe_status_reports_failure_without_raising(monkeypatch):
    """探测本身失败必须如实进消息,既不吞掉也不升级成另一个异常。"""
    from ashare_gauntlet.data import tushare_source

    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr("requests.post", boom)
    out = tushare_source._probe_status("https://x.example.com")
    assert "状态码探测失败" in out and "OSError" in out
