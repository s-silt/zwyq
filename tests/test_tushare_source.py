"""Tests for the Tushare connection helper.

The one behavior pinned here is the proxy bypass that the live debugging session
established: on Windows, popping the proxy *environment* variables is NOT enough,
because `requests` (and therefore tushare's internal `requests.post`) also reads
the system proxy from the registry. A local Clash proxy at 127.0.0.1:7891 then
hijacks the request to the domestic mirror IP and returns 502. The fix is to add
the mirror host to NO_PROXY so the request goes direct.
"""

import os
from functools import partial

import pandas as pd
import pytest

from ashare_gauntlet.data.tushare_source import (
    TushareDataUnavailable,
    TushareSDKIncompatible,
    _install_fail_loud_query,
    make_pro_api,
)


MIRROR = "https://mirror.example.com"


class _FakeDataApi:
    """最小 DataApi 替身:复刻 __getattr__ → partial(self.query, name) 的分发。"""

    def __init__(self, result, *, http_url=None):
        self._result = result
        self.calls = []
        self._DataApi__token = "SECRET-TOKEN-abc123"
        if http_url is not None:
            self._DataApi__http_url = http_url

    def query(self, api_name, fields="", **kwargs):
        self.calls.append((api_name, fields, kwargs))
        return self._result

    def __getattr__(self, name):
        # 只对接口名分发。私有/dunder 照常抛 AttributeError,否则 hasattr 恒为真,
        # "官方模式不得钉住 _DataApi__http_url" 这类断言会被悄悄废掉
        if name.startswith("_"):
            raise AttributeError(name)
        return partial(self.query, name)


def test_make_pro_api_without_mirror_preserves_sdk_defaults_and_proxy(monkeypatch):
    fake = _FakeDataApi(None)          # 替身须带 query:装配口会拒绝装不上保护的 client
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7891")
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setattr("ashare_gauntlet.data.tushare_source.ts.pro_api", lambda token: fake)

    assert make_pro_api("tok123", None) is fake
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7891"
    assert os.environ["NO_PROXY"] == "localhost"
    assert not hasattr(fake, "_DataApi__http_url")
    assert not hasattr(fake, "_DataApi__timeout")


def test_make_pro_api_blank_mirror_uses_official_mode(monkeypatch):
    fake = _FakeDataApi(None)
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

def test_swallowed_http_error_raises_instead_of_empty_frame():
    """零列空表=响应不可用,必须抛错;绝不能当成'今日无数据'返回给调用方。"""
    pro = _install_fail_loud_query(_FakeDataApi(pd.DataFrame(), http_url=MIRROR))

    with pytest.raises(TushareDataUnavailable) as exc:
        pro.stock_basic(list_status="L")

    msg = str(exc.value)
    assert "stock_basic" in msg          # 哪个接口
    assert "mirror.example.com" in msg   # 哪个端点
    # 不宣称具体 HTTP 状态码:零列有 HTTP 失败与空 schema 两种来源,单看 DataFrame
    # 分不出,断言"一定是 403"会超出证据(codex 复审 P2)
    assert "403" not in msg and "401" not in msg
    # 绝不泄漏 token 明文(出现"token 有效期"这类提示文案是允许的)
    assert "SECRET-TOKEN-abc123" not in msg


def test_genuine_empty_result_with_columns_passes_through():
    """真实的空结果(带列名)不能被误判为故障——非交易日/无记录是合法返回。"""
    empty_but_typed = pd.DataFrame(columns=["ts_code", "name", "industry"])
    pro = _install_fail_loud_query(_FakeDataApi(empty_but_typed, http_url=MIRROR))

    out = pro.stock_basic(list_status="L")
    assert list(out.columns) == ["ts_code", "name", "industry"]
    assert len(out) == 0


def test_normal_result_passes_through_untouched():
    df = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [12.3]})
    pro = _install_fail_loud_query(_FakeDataApi(df, http_url=MIRROR))
    out = pro.daily(trade_date="20260821")
    assert out.equals(df)


def test_wrapper_covers_every_endpoint_via_getattr_dispatch():
    """实例属性优先于类属性 → 所有接口名都走包装后的 query,不只 stock_basic。"""
    fake = _FakeDataApi(pd.DataFrame({"a": [1]}), http_url=MIRROR)
    pro = _install_fail_loud_query(fake)
    pro.daily(trade_date="20260821")
    pro.trade_cal(start_date="20260801")
    pro.adj_factor(trade_date="20260821")
    assert [c[0] for c in fake.calls] == ["daily", "trade_cal", "adj_factor"]


def test_client_without_query_fails_loud_not_silently_unprotected():
    """★没有 query 时必须抛错。原先是"原样返回",等于**悄悄关掉安全保护**——
    未来 SDK 改结构就会无声退回到"HTTP 错误变空表"的老毛病(codex 复审)。"""
    class Bare:
        pass

    with pytest.raises(TushareSDKIncompatible):
        _install_fail_loud_query(Bare())



# ── ★真入口锁:保护必须由 make_pro_api 这个唯一装配口装上 ──
#
# codex 复审 P2:上面那些用例直接调 _install_fail_loud_query,只锁住包装器内部接线。
# 把 make_pro_api 里的安装调用删掉,它们照样绿——那不是锁,是自证。
# 下面这条走真实 tushare.DataApi,只 mock requests.post 返回一个 bool 为 False 的
# Response(正是 403/401 的形状),官方与镜像两条分支都要触发保护。

@pytest.mark.parametrize("mirror", [None, "http://8.148.76.181:8686/"],
                         ids=["官方端点", "镜像端点"])
def test_make_pro_api_installs_protection_on_both_branches(monkeypatch, mirror):
    monkeypatch.setenv("NO_PROXY", "")

    class _FailedResponse:
        """复刻 requests.Response 在 4xx/5xx 下的关键行为:__bool__ 为 False。

        SDK 的 `if res:` 正是踩在这上面,把 HTTP 错误变成裸 pd.DataFrame()。
        """
        status_code = 403
        text = '{"code":0,"data":{"fields":[],"items":[]}}'

        def __bool__(self):
            return False

    monkeypatch.setattr("requests.post", lambda *a, **k: _FailedResponse())

    pro = make_pro_api("tok123", mirror)

    # 走真实 DataApi.__getattr__ 分发,不碰任何私有属性
    with pytest.raises(TushareDataUnavailable) as exc:
        pro.stock_basic(list_status="L")
    assert "stock_basic" in str(exc.value)

    # 核心 EOD 接口同样受保护(它们零列就是 schema 退化,必须 fail-loud)
    with pytest.raises(TushareDataUnavailable):
        pro.daily(trade_date="20260824")


def test_make_pro_api_success_path_unaffected_on_real_sdk(monkeypatch):
    """保护不能误伤正常响应:带列名的结果照常返回(含 0 行的合法空结果)。"""
    monkeypatch.setenv("NO_PROXY", "")

    class _OkResponse:
        text = '{"code":0,"data":{"fields":["ts_code","close"],"items":[]}}'

        def __bool__(self):
            return True

    monkeypatch.setattr("requests.post", lambda *a, **k: _OkResponse())

    pro = make_pro_api("tok123", "http://1.2.3.4:8080")
    out = pro.daily(trade_date="20260824")
    assert list(out.columns) == ["ts_code", "close"]
    assert len(out) == 0
