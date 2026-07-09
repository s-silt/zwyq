"""盘中价格哨兵的数据层 —— 腾讯行情公开接口(延迟秒-分钟级),零新依赖(urllib)。

来源评估(stock-sdk 研读,2026-07-08):该库全部能力里唯一的真空白是盘中实时价
(全链路 EOD,盯盘 routine 收盘后才跑);其余(技术指标/筹码/玩具回测)不吸纳。
引入方式取最小依赖:不引 JS 库/不接其 MCP,直接打它底层用的同一个公开接口。

**口径隔离(数据源纯净审计线)**:盘中价只服务提醒,绝不写入 data/cache 研究缓存
——与 tushare EOD 的复权/口径体系不混;历史研究一律走 EOD 数据。
接口格式:GET http://qt.gtimg.cn/q=sh600875,sz000589 → GBK 文本,每行
v_sh600875="1~东方电气~600875~现价~昨收~今开~...~涨跌幅[32]~...";
"""
from __future__ import annotations

import urllib.request

_FIELD_NAME, _FIELD_CODE, _FIELD_LAST, _FIELD_PREV, _FIELD_PCT = 1, 2, 3, 4, 32


def tencent_symbol(ts_code: str) -> str:
    """ts_code(600875.SH)→ 腾讯符号(sh600875)。非标准 ts_code fail-loud,防静默查错票。"""
    if "." not in ts_code:
        raise ValueError(f"需要 ts_code 格式(如 600875.SH),得到 {ts_code!r}")
    code, exch = ts_code.split(".", 1)
    if exch.upper() not in ("SH", "SZ") or not code.isdigit():
        raise ValueError(f"仅支持沪深 ts_code,得到 {ts_code!r}")
    return exch.lower() + code


def parse_tencent_quote(text: str) -> dict[str, dict]:
    """腾讯行情响应 → {ts_code: {name,last,prev_close,pct}}。

    空响应/无有效行 fail-loud:哨兵沉默比误报危险(接口挂了要知道,不能当"无事")。
    垃圾行(v_pv_none 等空值行)跳过。
    """
    out: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("v_") or '="' not in line:
            continue
        sym = line[2:line.index('="')]
        body = line[line.index('="') + 2:].rstrip('";')
        parts = body.split("~")
        if len(parts) <= _FIELD_PCT or not parts[_FIELD_LAST]:
            continue
        if sym[:2] not in ("sh", "sz"):
            continue
        try:   # 单行字段异常('--'/空串)跳过该行,不崩整批(哨兵要报出其余持仓)
            out[f"{parts[_FIELD_CODE]}.{sym[:2].upper()}"] = {
                "name": parts[_FIELD_NAME],
                "last": float(parts[_FIELD_LAST]),
                "prev_close": float(parts[_FIELD_PREV]),
                "pct": float(parts[_FIELD_PCT]),
            }
        except ValueError:
            continue
    if not out:
        raise ValueError("腾讯行情响应无有效行——接口异常或符号全错,拒绝静默当作无行情")
    return out


def fetch_quotes(ts_codes: list[str], timeout: float = 10.0) -> dict[str, dict]:
    """批量拉实时价(一次 GET)。网络失败向上抛(操作工具,fail-loud 由调用方重试)。"""
    syms = ",".join(tencent_symbol(c) for c in ts_codes)
    req = urllib.request.Request(f"http://qt.gtimg.cn/q={syms}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_tencent_quote(resp.read().decode("gbk", errors="replace"))


def alert_level(last: float, stop: "float | None", warn_dist: float,
                band: "tuple[float, float] | None" = None) -> str:
    """警报分级:BREACH(≤止损)> NEAR(距止损<warn_dist)> BAND(触发带命中)> OK。

    warn_dist 是操作性提醒参数(CLI 可调,默认见调用方),非评分常数;
    band=观察名单的买入触发带 [low, high],命中报 BAND。
    """
    if stop is not None:
        if last <= stop:
            return "BREACH"
        if last / stop - 1.0 < warn_dist:
            return "NEAR"
    if band is not None and band[0] <= last <= band[1]:
        return "BAND"
    return "OK"
