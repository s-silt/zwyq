"""entry_check —— 单票右侧确认 + 整手仓位建议(双仓制执行层 CLI)。

把 memory trading-constraints 里已认可的双仓制翻译成可执行检查:
- 短线仓(short):单笔风险 = 账户 1%,硬止损 -7%(制度参数,出处 memory
  trading-constraints:"短线仓1只≤1万,硬止损-7%,单笔风险预算=账户1%");
- 长线仓(long):单笔风险 = 账户 1.5%,止损 -13%(制度区间 -12%~-15% 取中值,
  出处同上:"长线仓止损-12~-15%或逻辑坏")。

右侧判据(execution.entry_readiness,全定义性比较):缩量企稳 + 收复5日线;
⚡ 若近5日触及涨停(factor_model.touched_limit_up,窗口约定同库内"近5日"),
打印"涨停顶上来的,短线默认不追"警示(铁律出处 memory momentum-screen-limitup)。

数据全部走本地缓存(data/cache 的 daily / adj_factor / stk_limit;stk_limit 缺日
才会经 fetch_market_day 打 API 补拉)。判定用前复权价(close×adj_factor,消除窗口内
分红/送转跳空);下单报价与止损用最新**真实收盘价**。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.entry_check \
           --code 600989.SH [--bucket short|long] [--account 75000]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import fetch_market_day
from ashare_gauntlet.data.partition import date_partition_files
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.execution import MIN_BARS, entry_readiness, position_size
from ashare_gauntlet.factor_model import touched_limit_up

CACHE = "data/cache"

# 涨停检查窗口 = 近5日:复用库内既有"近5日触及涨停"窗口约定(touched_limit_up 的
# 使用场景,见 memory momentum-screen-limitup"新强名先问是不是涨停顶上来的")
LIMIT_WINDOW = 5

# 双仓制制度参数(用户已认可,出处 memory trading-constraints;非库内可调常数):
#   risk_pct —— 单笔风险占账户比例;stop_frac —— 止损距离(入场价的跌幅)
#   cap —— 单票市值上限:短线仓"1只≤1万"是制度明文(同出处);长线无单票市值上限
BUCKETS: dict[str, dict] = {
    "short": {"risk_pct": 0.01, "stop_frac": 0.07, "cap": 10000.0,
              "note": "短线仓:风险=账户1%,硬止损-7%(memory trading-constraints)"},
    "long": {"risk_pct": 0.015, "stop_frac": 0.13, "cap": None,
             "note": "长线仓:风险=账户1.5%,止损-13%(制度区间-12%~-15%取中值,"
                     "memory trading-constraints)"},
}

# 默认账户市值:用户当前预算口径(memory trading-constraints,2026-06 起 6万+已加仓),
# 是用户参数不是库常数 —— 实际以 --account 覆盖为准
DEFAULT_ACCOUNT = 75000.0


def _make_pro() -> object:
    load_env_local()
    return make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])


def _trade_dates(cache_dir: str) -> list[str]:
    """本地 daily 缓存的交易日历(日分区文件名即交易日),升序。空缓存 fail-loud。

    走 date_partition_files:daily/ 实际混入过整段拉取的 <code>_<start>_<end>.parquet,
    直接 glob 会把污染文件名前 8 位当"交易日"(见 ashare_gauntlet.data.partition)。
    """
    days = [os.path.basename(f)[:8] for f in date_partition_files(cache_dir, "daily")]
    if not days:
        raise SystemExit(f"entry_check: {cache_dir}/daily 无缓存 —— 先跑 scripts.refresh 回填")
    return days


def load_code_history(code: str, cache_dir: str, min_bars: int) -> pd.DataFrame:
    """从最新交易日倒读 daily+adj_factor,凑够该票 min_bars 根有效K线即停(纯 IO 瘦身)。

    停牌日该票无行、自然跳过;读穿全部缓存仍不足 min_bars → fail-loud(新上市/长停,
    拒绝在残缺数据上判定)。返回按 trade_date 升序的 close/vol/adj_close。
    """
    rows: list[pd.DataFrame] = []
    for day in reversed(_trade_dates(cache_dir)):
        d = pd.read_parquet(Path(cache_dir) / "daily" / f"{day}.parquet",
                            columns=["ts_code", "trade_date", "close", "vol"])
        r = d[d["ts_code"] == code]
        if r.empty:
            continue  # 停牌/未上市日
        a = pd.read_parquet(Path(cache_dir) / "adj_factor" / f"{day}.parquet",
                            columns=["ts_code", "trade_date", "adj_factor"])
        rows.append(r.merge(a, on=["ts_code", "trade_date"], how="left"))
        if len(rows) >= min_bars:
            break
    if len(rows) < min_bars:
        raise SystemExit(
            f"entry_check: {code} 本地缓存仅 {len(rows)} 根有效K线(< {min_bars})"
            f"—— 新上市/长期停牌/代码有误,拒绝判定")
    df = pd.concat(rows, ignore_index=True).sort_values("trade_date").reset_index(drop=True)
    if df["adj_factor"].isna().any():
        raise SystemExit(f"entry_check: {code} 有交易日缺 adj_factor —— 缓存不齐,先补拉再判定")
    df["adj_close"] = df["close"].astype(float) * df["adj_factor"].astype(float)
    return df


def limit_up_recent(code: str, cache_dir: str, pro: object) -> bool:
    """该票近 LIMIT_WINDOW 个交易日是否触及过涨停(daily.high ≥ stk_limit.up_limit)。"""
    window = _trade_dates(cache_dir)[-LIMIT_WINDOW:]
    daily_5d = pd.concat([fetch_market_day(pro, "daily", d, cache_dir) for d in window],
                         ignore_index=True)
    stk_limit_5d = pd.concat([fetch_market_day(pro, "stk_limit", d, cache_dir) for d in window],
                             ignore_index=True)
    return code in touched_limit_up(daily_5d, stk_limit_5d)


def main() -> None:
    ap = argparse.ArgumentParser(description="单票右侧确认 + 整手仓位建议(双仓制执行层)")
    ap.add_argument("--code", required=True, help="ts_code,如 600989.SH")
    ap.add_argument("--bucket", choices=sorted(BUCKETS), default="short",
                    help="仓别:short=短线仓(默认)/ long=长线仓")
    ap.add_argument("--account", type=float, default=DEFAULT_ACCOUNT,
                    help=f"账户市值(默认 {DEFAULT_ACCOUNT:.0f},用户参数请按实际覆盖)")
    args = ap.parse_args()

    hist = load_code_history(args.code, CACHE, MIN_BARS)
    as_of = str(hist["trade_date"].iloc[-1])
    er = entry_readiness(hist["adj_close"], hist["vol"])

    bucket = BUCKETS[args.bucket]
    entry = float(hist["close"].iloc[-1])          # 下单口径:最新真实收盘价(非复权价)
    stop = entry * (1.0 - bucket["stop_frac"])
    ps = position_size(args.account, bucket["risk_pct"], entry, stop)

    print(f"=== entry_check {args.code}(截至 {as_of},判定用前复权)===")
    print(f"右侧判定: {er['label']}")
    print(f"  收复5日线: {'✓' if er['above_ma5'] else '✗'}"
          f"(收盘 {er['close']:.2f} vs MA5 {er['ma5']:.2f})")
    print(f"  缩量:     {'✓' if er['shrinking_vol'] else '✗'}"
          f"(最新量 {er['vol']:.0f} 手 vs 前5日均量 {er['vol_ma5']:.0f} 手)")

    if limit_up_recent(args.code, CACHE, _make_pro()):
        print(f"⚡ 近{LIMIT_WINDOW}日触及涨停 —— 涨停顶上来的,短线默认不追"
              f"(memory momentum-screen-limitup)")

    risk_budget = args.account * bucket["risk_pct"]
    print(f"\n仓位建议({args.bucket};{bucket['note']}):")
    print(f"  账户 {args.account:,.0f} × 风险 {bucket['risk_pct']:.1%} = 预算 {risk_budget:,.0f} 元")
    print(f"  入场 {entry:.2f}(最新收盘) | 止损 {stop:.2f}(-{bucket['stop_frac']:.0%})"
          f" | 单股风险 {entry - stop:.2f}")
    if ps["shares"] == 0:
        print("  → 风险预算不足一手的止损风险,建议 0 手(不硬凑一手突破风险预算)")
    else:
        print(f"  → 建议 {ps['lots']} 手({ps['shares']} 股) | 成本 {ps['cost']:,.0f} 元"
              f" | 触止损最大亏 {ps['max_loss']:,.0f} 元(≤ 预算 {risk_budget:,.0f})")
        cap = bucket["cap"]
        if cap is not None and ps["cost"] > cap:
            # 双约束取更紧者:风险公式给的手数触碰单票市值上限 → 按上限降手
            capped_lots = int(cap // (entry * 100))  # 100 = A股一手(交易所规则)
            print(f"  ⚠ 成本 {ps['cost']:,.0f} 超单票上限 {cap:,.0f}(制度:短线仓1只≤1万,"
                  f"memory trading-constraints)→ 按上限降为 {capped_lots} 手"
                  f"({capped_lots * 100} 股,成本 {capped_lots * 100 * entry:,.0f})")


if __name__ == "__main__":
    main()
