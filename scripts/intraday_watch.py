"""盘中价格哨兵 —— 持仓止损距离 + 观察名单触发带,一条命令随手跑。

用法: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.intraday_watch [--warn-dist 0.03]

读 data/holdings.json(持仓+止损)与 data/trigger_bands.json(观察触发带;注意
data/watchlist.json 是 6/19 种子名单另有消费者,勿混用),拉腾讯实时价
(延迟秒-分钟级),按严重度输出:🔴 BREACH(≤止损,按纪律执行)> 🟠 NEAR(逼近)>
🟡 BAND(触发带命中,查入场条件)> 静默。盘中价不入研究缓存(口径隔离)。
--warn-dist 为操作性提醒参数(默认 0.03,可调),非评分常数。
非交易时段跑=显示最近收盘价,警报语义不变。
"""
from __future__ import annotations

import argparse
import json

from datetime import datetime
from zoneinfo import ZoneInfo

from ashare_gauntlet.config import (
    HOLDINGS_PATH as HOLDINGS,
    INTRADAY_STATE_PATH as STATE,  # 定时模式的上次警报态(去重用,非研究数据)
    TRIGGER_BANDS_PATH as WATCHLIST,
)
from ashare_gauntlet.account_state import normalize_bucket
from ashare_gauntlet.intraday import alert_level, fetch_quotes, sentinel_delta

TP_MULT = 1.25   # 长线止盈提示线=成本×1.25(双仓制"+25%减半锁利"既有约定,非新常数)
_ICON = {"BREACH": "🔴", "NEAR": "🟠", "PROFIT": "🟢", "BAND": "🟡", "OK": "  "}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warn-dist", type=float, default=0.03, dest="warn_dist",
                    help="距止损低于该比例即 🟠 提醒(操作参数,默认3%%)")
    ap.add_argument("--dedupe", action="store_true",
                    help="定时任务模式:当日已报最高级封存(latch)去重+解除单报+交易时段守卫;"
                         "状态存 data/intraday_alert_state.json;手动跑不带此参=全量输出不碰状态")
    a = ap.parse_args(argv)

    now = datetime.now(ZoneInfo("Asia/Shanghai"))   # 显式北京时区(部署机时区无关)
    today = now.strftime("%Y%m%d")
    if a.dedupe:
        # 交易时段守卫(确定性逻辑):9:25-11:35 / 12:55-15:05,周末退出;
        # 法定节假日(工作日)会漏过守卫:行情为静态收盘价,当日 latch 后每假日至多重报一次
        # ——已知小噪声,接交易日历属过度工程(EOD routine 已有日历)
        hm = now.hour * 100 + now.minute
        if now.weekday() >= 5 or not ((925 <= hm <= 1135) or (1255 <= hm <= 1505)):
            print("哨兵(定时):非交易时段,静默。")
            return

    hold = json.load(open(HOLDINGS, encoding="utf-8"))
    try:
        watch = json.load(open(WATCHLIST, encoding="utf-8"))["items"]
    except FileNotFoundError:
        watch = []
    codes = [p["ts_code"] for p in hold["positions"]] + [w["ts_code"] for w in watch]
    q = fetch_quotes(codes)

    rows: list[tuple] = []                     # (sev, key|None, line);dedupe 过滤按 key
    levels: dict[str, str] = {}
    fps: dict[str, str] = {}
    names: dict[str, str] = {}
    if hold.get("as_of") and hold["as_of"] < today:
        # 陈旧持仓=独立操作风险,走状态机合成键(每日一报,不随价格警报静默而丢失)
        k = "meta:holdings_stale"
        levels[k] = "NEAR"; fps[k] = hold["as_of"]; names[k] = "持仓数据"
        rows.append((9, k, f"⚠ holdings.json 口径日期 {hold['as_of']}(非今日)——持仓若有变动请先发截图,"
                           f"以下警报可能基于旧仓"))
    for p in hold["positions"]:
        r = q.get(p["ts_code"])
        if r is None:                          # 缺行情=运营警报合成键,不许被静默分支吞掉
            k = f"miss:{p['ts_code']}"
            levels[k] = "NEAR"; fps[k] = "pos"; names[k] = p["name"]
            rows.append((2, k, f"⚠ {p['name']} 无行情返回(符号/停牌/退市?)"))
            continue
        # 无止损价=该仓完全没有 BREACH/NEAR 保护(alert_level 会跳过止损分支并返回 OK),
        # 若不合成警报键,哨兵会对一只裸仓打印"全部安静"/在 --dedupe 下整行过滤 ——
        # 把"没有止损"当成"没有风险",违反"缺失不得解释为安全"(跨层审计 P1)
        stop_px = p.get("stop")
        if not (isinstance(stop_px, (int, float)) and not isinstance(stop_px, bool)
                and stop_px > 0):
            k = f"nostop:{p['ts_code']}"
            levels[k] = "NEAR"; fps[k] = f"{p['shares']}@{p['cost']}@nostop"; names[k] = p["name"]
            rows.append((2, k, f"⚠ {p['name']} 未设止损价——本仓无 BREACH/NEAR 保护,补 stop"))
        # 归一为 None 再传:stop=0/负/字符串会让 alert_level 除零或类型错,一只脏值
        # 就炸掉整轮哨兵(对抗复核 P2)。脏值与未填同义=无保护,走无止损分支。
        stop_arg = stop_px if (isinstance(stop_px, (int, float))
                               and not isinstance(stop_px, bool) and stop_px > 0) else None
        # 经权威归一再比较:英文 "long" 仓过去拿不到 +25% 止盈提示(跨层审计 P1)
        tp = p["cost"] * TP_MULT if normalize_bucket(p.get("bucket")) == "long" else None
        lvl = alert_level(r["last"], stop_arg, a.warn_dist, tp=tp)
        key = f"pos:{p['ts_code']}"            # 命名空间键:与观察名单同代码不互相覆盖
        levels[key] = lvl
        fps[key] = f"{p['shares']}@{p['cost']}@{p.get('stop')}"   # 持仓指纹:变化即重置 latch
        names[key] = p["name"]
        pnl = (r["last"] / p["cost"] - 1) * 100
        dist = (r["last"] / p["stop"] - 1) * 100 if p.get("stop") else float("nan")
        sev = {"BREACH": 3, "NEAR": 2, "PROFIT": 2, "BAND": 1, "OK": 0}[lvl]
        tail = {"BREACH": "  ← 按纪律执行", "PROFIT": "  ← 达成本+25%,提示减半锁利(既有约定)"}.get(lvl, "")
        rows.append((sev, key, f"{_ICON[lvl]} {p['name']:　<5} {r['last']:>8.2f} 今{r['pct']:+5.2f}% "
                               f"浮盈{pnl:+6.1f}% 距止损{dist:+5.1f}%{tail}"))
    for w in watch:
        r = q.get(w["ts_code"])
        if r is None:
            k = f"miss:{w['ts_code']}"
            levels[k] = "NEAR"; fps[k] = "watch"; names[k] = w["name"]
            rows.append((2, k, f"⚠ {w['name']} [观察] 无行情返回(符号/停牌/退市?)"))
            continue
        band = (w["band_low"], w["band_high"]) if w.get("band_low") is not None else None
        lvl = alert_level(r["last"], None, a.warn_dist, band=band)
        key = f"watch:{w['ts_code']}"
        levels[key] = lvl
        fps[key] = f"{w.get('band_low')}-{w.get('band_high')}"    # 带指纹:EOD 改带=新观察条件
        names[key] = w["name"]
        sev = 1 if lvl == "BAND" else 0
        rows.append((sev, key, f"{_ICON[lvl]} {w['name']:　<5} {r['last']:>8.2f} 今{r['pct']:+5.2f}% "
                               f"[观察] {w.get('note', '')}" + ("  ← 带内,查入场条件" if lvl == "BAND" else "")))

    if a.dedupe:   # 定时模式:latch 去重;行过滤严格按 key(不做名字子串匹配)
        try:
            prev = json.load(open(STATE, encoding="utf-8"))
        except FileNotFoundError:
            prev = {}
        if prev and "date" not in prev:        # 旧平面 schema:显式重置一次并说明,不静默假装兼容
            print("(状态文件旧版已重置,本次可能重报一轮既有警报)")
            prev = {}
        report, cleared_keys, new_state = sentinel_delta(prev, levels, today, fps=fps)
        json.dump(new_state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for k in cleared_keys:
            rows.append((2, k, f"🟢 {names.get(k, k)} 破线警报解除(回到止损线上方)"
                               f"——若已按纪律卖出请发截图更新持仓"))
        keep = set(report) | set(cleared_keys)
        rows = [t for t in rows if t[1] in keep]
        if not rows:
            print("哨兵(定时):警报态无变化,静默。")
            return

    rows.sort(key=lambda x: -x[0])
    print("=== 盘中哨兵(腾讯实时价,延迟秒-分钟级;盘外=最近收盘)===")
    for _, _k, line in rows:
        print(line)
    n_alert = sum(1 for s, _k, _l in rows if s > 0)
    print(f"—— {'⚠ ' + str(n_alert) + ' 项需要注意' if n_alert else '全部安静'};"
          f"盘中价仅提醒用,不入研究缓存")


if __name__ == "__main__":
    main()
