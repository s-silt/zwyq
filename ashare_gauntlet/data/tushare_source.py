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


class TushareTransportError(RuntimeError):
    """HTTP 层失败被 SDK 吞成空表时抛出。

    区分依据是 SDK 自身的两条返回路径:成功走 ``pd.DataFrame(items, columns=fields)``
    —— 即便 0 行也**带列名**;HTTP 失败走裸 ``pd.DataFrame()`` —— **零列**。
    因此"零列 DataFrame"是传输失败的确定签名,不会把真实的空结果误判成故障。
    """

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def _probe_status(http_url: str, timeout: int = 10) -> str:
    """尽力探一次 HTTP 状态码,只为把错误消息变得可行动(403=授权/500=服务端)。

    探测本身失败不改变结论——调用方无论如何都会抛 TushareTransportError,
    这里只把"探不到"如实写进消息,不吞掉、也不升级为新的异常。
    """
    try:
        import requests
        resp = requests.post(http_url, json={"api_name": "__probe__"}, timeout=timeout)
        return f"HTTP {resp.status_code}"
    except Exception as exc:            # noqa: BLE001 —— 诊断增强,结论不依赖它
        return f"状态码探测失败({type(exc).__name__})"


def _install_fail_loud_query(pro):
    """把 SDK 吞 HTTP 错误的 query 换成 fail-loud 版本。

    DataApi.__getattr__ 走 ``partial(self.query, name)``,而实例属性优先于类属性,
    因此挂一个同名实例属性即可覆盖所有接口(daily/stock_basic/trade_cal ...)。
    没有 query 属性的对象(测试替身)原样返回,不制造新的失败模式。
    """
    original = getattr(pro, "query", None)
    if not callable(original):
        return pro

    def query(api_name, fields="", **kwargs):
        result = original(api_name, fields=fields, **kwargs)
        if isinstance(result, pd.DataFrame) and len(result.columns) == 0:
            url = getattr(pro, "_DataApi__http_url", "") or "<SDK 默认端点>"
            host = urlparse(url).hostname or url
            raise TushareTransportError(
                f"tushare 接口 {api_name!r} 请求失败被 SDK 吞成空表"
                f"(零列 DataFrame=传输失败签名,非真实空结果);"
                f"端点 {host} {_probe_status(url)}。"
                "拒绝把请求失败当作'今日无数据'——请检查镜像订阅/IP 授权/token 有效期"
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
