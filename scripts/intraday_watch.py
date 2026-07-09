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

from ashare_gauntlet.intraday import alert_level, fetch_quotes

HOLDINGS = "data/holdings.json"
WATCHLIST = "data/trigger_bands.json"
_ICON = {"BREACH": "🔴", "NEAR": "🟠", "BAND": "🟡", "OK": "  "}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warn-dist", type=float, default=0.03, dest="warn_dist",
                    help="距止损低于该比例即 🟠 提醒(操作参数,默认3%%)")
    a = ap.parse_args(argv)

    hold = json.load(open(HOLDINGS, encoding="utf-8"))
    try:
        watch = json.load(open(WATCHLIST, encoding="utf-8"))["items"]
    except FileNotFoundError:
        watch = []
    codes = [p["ts_code"] for p in hold["positions"]] + [w["ts_code"] for w in watch]
    q = fetch_quotes(codes)

    rows = []
    for p in hold["positions"]:
        r = q.get(p["ts_code"])
        if r is None:
            rows.append((0, f"⚠ {p['name']} 无行情返回(符号/停牌?)"))
            continue
        lvl = alert_level(r["last"], p.get("stop"), a.warn_dist)
        pnl = (r["last"] / p["cost"] - 1) * 100
        dist = (r["last"] / p["stop"] - 1) * 100 if p.get("stop") else float("nan")
        sev = {"BREACH": 3, "NEAR": 2, "BAND": 1, "OK": 0}[lvl]
        rows.append((sev, f"{_ICON[lvl]} {p['name']:　<5} {r['last']:>8.2f} 今{r['pct']:+5.2f}% "
                          f"浮盈{pnl:+6.1f}% 距止损{dist:+5.1f}%"
                          + ("  ← 按纪律执行" if lvl == "BREACH" else "")))
    for w in watch:
        r = q.get(w["ts_code"])
        if r is None:   # 缺行情同样 surface(退市/符号错/字段异常'--'),不静默跳过
            rows.append((1, f"⚠ {w['name']} [观察] 无行情返回(符号/停牌/退市?)"))
            continue
        band = (w["band_low"], w["band_high"]) if w.get("band_low") is not None else None
        lvl = alert_level(r["last"], None, a.warn_dist, band=band)
        sev = 1 if lvl == "BAND" else 0
        rows.append((sev, f"{_ICON[lvl]} {w['name']:　<5} {r['last']:>8.2f} 今{r['pct']:+5.2f}% "
                          f"[观察] {w.get('note', '')}" + ("  ← 带内,查入场条件" if lvl == "BAND" else "")))

    rows.sort(key=lambda x: -x[0])
    print("=== 盘中哨兵(腾讯实时价,延迟秒-分钟级;盘外=最近收盘)===")
    for _, line in rows:
        print(line)
    n_alert = sum(1 for s, _ in rows if s > 0)
    print(f"—— {'⚠ ' + str(n_alert) + ' 项需要注意' if n_alert else '全部安静'};"
          f"盘中价仅提醒用,不入研究缓存")


if __name__ == "__main__":
    main()
