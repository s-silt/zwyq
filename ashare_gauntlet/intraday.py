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

import re
import urllib.parse
import urllib.request
from itertools import islice
from typing import Iterator

_FIELD_NAME, _FIELD_CODE, _FIELD_LAST, _FIELD_PREV, _FIELD_PCT = 1, 2, 3, 4, 32
_RECORD = re.compile(r'v_(\w+)="([^"]*)"')   # 记录级正则:换行/分号两种物理分隔都覆盖
# 单批符号上限(stock-sdk MAX_BATCH_SIZE 同款;URL 超长脆断的操作参数,非评分常数)
BATCH = 500


def chunked(items: list, size: int) -> "Iterator[list]":
    """列表按 size 切片(批量请求用)。"""
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


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
    for sym, body in _RECORD.findall(text):
        # 正则按记录提取(stock-sdk parser 同语义):单行多记录(分号分隔)不再串字段;
        # v_pv_none_match 空壳(请求了不存在的符号)与非沪深前缀在此过滤
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
    """批量拉实时价(HTTPS+URL encode+超上限自动切片)。网络失败向上抛(调用方重试)。

    缺失票(请求了但响应无该记录:退市/符号错/字段异常'--')**不在返回 dict 里**,
    由调用方逐票 surface(intraday_watch 对持仓打 ⚠)——静默遗漏是哨兵最危险的失败模式。
    """
    out: dict[str, dict] = {}
    for batch in chunked(ts_codes, BATCH):
        syms = urllib.parse.quote(",".join(tencent_symbol(c) for c in batch), safe=",")
        req = urllib.request.Request(f"https://qt.gtimg.cn/q={syms}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out.update(parse_tencent_quote(resp.read().decode("gbk", errors="replace")))
    return out


def alert_level(last: float, stop: "float | None", warn_dist: float,
                band: "tuple[float, float] | None" = None,
                tp: "float | None" = None) -> str:
    """警报分级:BREACH(≤止损)> NEAR(距止损<warn_dist)> PROFIT(≥止盈提示线)
    > BAND(触发带命中)> OK。

    warn_dist 是操作性提醒参数(CLI 可调),非评分常数;band=观察名单买入触发带;
    tp=止盈提示线(长线=成本×1.25,出处:双仓制"+25% 减半锁利"既有约定)——
    提示减半锁利而非清仓,与"财报季持有"不冲突;止损类警报优先(风险先于收益)。
    """
    if stop is not None:
        if last <= stop:
            return "BREACH"
        if last / stop - 1.0 < warn_dist:
            return "NEAR"
    if tp is not None and last >= tp:
        return "PROFIT"
    if band is not None and band[0] <= last <= band[1]:
        return "BAND"
    return "OK"


_SEVERITY = {"OK": 0, "BAND": 1, "PROFIT": 2, "NEAR": 2, "BREACH": 3}


def sentinel_delta(prev_state: dict, cur: dict[str, str], trade_date: str,
                   fps: "dict[str, str] | None" = None) -> tuple[list, list, dict]:
    """定时哨兵状态机(设计审查后 v2)。返回 (要报的键, 解除要报的键, 新状态)。

    语义(每条对应一个设计审查发现):
    - 键带命名空间(pos:/watch:)——同一 ts_code 在持仓与观察名单并存时不互相覆盖;
    - **当日已报最高级封存(latch)**:BREACH↔NEAR 贴线震荡不反复轰炸,同级/降级静默;
    - BREACH 解除(回到 OK/PROFIT/BAND)报一次并打 cleared 标,不重复;
    - 跨交易日重置:latch 只在当日有效(PROFIT/BREACH 每日至多一报,收盘后翻篇);
    - 持仓指纹(fps:如 股数@成本@止损)变化 → 该键重置(卖出再买不继承旧 latch)。
    状态结构:{"date": 交易日, "keys": {key: {"max": 已报最高级, "cleared": bool, "fp": 指纹}}}。
    """
    fps = fps or {}
    if prev_state.get("date") != trade_date:
        prev_keys: dict = {}
    else:
        prev_keys = dict(prev_state.get("keys", {}))
    report, cleared = [], []
    new_keys: dict = {}
    for key, lvl in cur.items():
        rec = prev_keys.get(key)
        if rec is not None and fps.get(key) is not None and rec.get("fp") not in (None, fps.get(key)):
            rec = None                       # 指纹变化=新仓,重置
        prev_max = rec.get("max", "OK") if rec else "OK"
        was_cleared = rec.get("cleared", False) if rec else False
        sev, prev_sev = _SEVERITY.get(lvl, 0), _SEVERITY.get(prev_max, 0)
        if prev_max == "BREACH" and was_cleared and lvl == "BREACH":
            # 解除后同日**再破线必须重报**(用户最后看到的是"安全",静默=危险漏报),
            # 重报后重新 latch(cleared=False),贴线震荡仍只报这一次
            report.append(key)
            new_keys[key] = {"max": "BREACH", "cleared": False, "fp": fps.get(key)}
        elif sev > prev_sev and sev > 0:
            report.append(key)
            new_keys[key] = {"max": lvl, "cleared": False, "fp": fps.get(key)}
        elif prev_max == "BREACH" and lvl in ("OK", "PROFIT", "BAND") and not was_cleared:
            # 解除=离开风险区(OK/PROFIT/BAND 都算,不只 OK——破线后直接反弹过止盈线同样是解除)
            cleared.append(key)
            new_keys[key] = {"max": prev_max, "cleared": True, "fp": fps.get(key)}
        else:
            new_keys[key] = {"max": prev_max if prev_sev >= sev else lvl,
                             "cleared": was_cleared, "fp": fps.get(key)}
    return report, cleared, {"date": trade_date, "keys": new_keys}
