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
3. tushare 的 SDK 把 **任何 HTTP 错误吞成空 DataFrame**:其 ``query`` 写的是
   ``res = requests.post(...); if res: ... else: return pd.DataFrame()``,而
   ``Response.__bool__`` 就是 ``res.ok`` —— 403/500/超时全部落进 else 分支,
   调用方拿到的是"今天没有数据"而不是"请求失败"。这直接违反项目的 fail-loud
   纪律(未知不得解释为安全),实测已导致 ``stock_basic`` 静默返回空表、
   镜像 403 却看不出任何异常。这里在 ``make_pro_api`` 这个唯一装配口把它拦回来。
"""

import os
from urllib.parse import urlparse

import pandas as pd
import tushare as ts


class TushareDataUnavailable(RuntimeError):
    """SDK 返回**零列** DataFrame——响应不可用,不得当作"今日无数据"。

    命名刻意不叫 TransportError:零列有两个来源,单看 DataFrame 分不出——
    ①HTTP 失败(403/401/500)走 SDK 的 ``else: return pd.DataFrame()``;
    ②服务端返回合法但无 schema 的 ``{"code":0,"data":{"fields":[],"items":[]}}``。
    两者都意味着这次调用**没拿到可用数据**,都该 fail-loud;但断言"一定是 403"
    会超出证据(codex 复审 P2)。故这里只陈述"零列/响应不可用",不编造状态码。

    对四个核心 EOD 接口(daily/adj_factor/daily_basic/stk_limit)而言,零列本就
    违反 schema 契约,抛错是唯一正确行为(CLAUDE.md:退化 schema 必须 fail-loud)。
    """


class TushareSDKIncompatible(RuntimeError):
    """装配口拿到的 client 没有可调用的 query——保护装不上,拒绝返回裸 client。

    静默放行等于**悄悄关掉安全保护**,正是本次要修的那类错误(codex 复审)。
    """

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def _install_fail_loud_query(pro):
    """把 SDK 吞 HTTP 错误的 query 换成 fail-loud 版本。

    DataApi.__getattr__ 走 ``partial(self.query, name)``,而实例属性优先于类属性,
    因此挂一个同名实例属性即可覆盖所有接口(daily/stock_basic/trade_cal ...)。
    没有 query 属性的对象(测试替身)原样返回,不制造新的失败模式。
    """
    original = getattr(pro, "query", None)
    if not callable(original):
        raise TushareSDKIncompatible(
            f"tushare client({type(pro).__name__})没有可调用的 query——"
            "fail-loud 包装装不上;静默放行等于关掉保护,拒绝返回裸 client")

    def query(api_name, fields="", **kwargs):
        result = original(api_name, fields=fields, **kwargs)
        if isinstance(result, pd.DataFrame) and len(result.columns) == 0:
            url = getattr(pro, "_DataApi__http_url", "") or "<SDK 默认端点>"
            host = urlparse(url).hostname or url
            raise TushareDataUnavailable(
                f"tushare 接口 {api_name!r} 返回零列 DataFrame(端点 {host})——"
                "响应不可用:HTTP 错误被 SDK 吞成空表,或服务端返回了无 schema 的空结果。"
                "拒绝当作'今日无数据'——请检查镜像订阅/IP 授权/token 有效期"
            )
        return result

    pro.query = query
    return pro


def make_pro_api(token: str, http_url: str | None = None, timeout: int = 120):
    """Build a Tushare client, optionally pinned directly to a mirror.

    With no explicit URL, the SDK's official endpoint and the caller's proxy
    environment are left untouched.  Mirror mode retains the Windows direct-
    connection workaround and the generous timeout needed by full-market pulls.
    """
    mirror_url = http_url.strip() if http_url else ""
    if not mirror_url:
        return _install_fail_loud_query(ts.pro_api(token))

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
    return _install_fail_loud_query(pro)
