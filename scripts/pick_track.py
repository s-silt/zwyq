"""pick_track —— 筛选器命中率闭环(纯测量,不打分、不改排序)。

审计点名的最大验证缺口:回测只有 N=40 单一 regime,factor_rank 的样本外有效性只能靠
"记录每期 D10 → 追踪后续真实收益" 积累证据;此前快照一直在攒(data/holdscore/*_factor.json)
但零脚本回读——筛选器实盘命中率处于零证据状态。本脚本补上这个闭环:

① diff:最近两期 factor json 的 D10 进/出名单(**新进票**正是动量陷阱高危,呼应
   memory momentum-screen-limitup"新强名先问是不是涨停顶上来的");
② 前向收益:对每期历史快照的 D10,算 快照日→今天 的前复权收益,对比全主板等权基准
   (超额 = 命中率证据;纯测量,几个月后回答"这筛选器到底选得准不准");
③ 沪深300 双基准 + regime 读数:宇宙内等权基准量"选股能力",沪深300 量"配置效果"
   (两者可背离:全主板齐跌时跑赢宇宙仍可能跑输 300);最近 20 交易日 300 涨跌
   帮助解读 D10 跑输是 α(选股)问题还是 β(市场)问题。
④ 成本后超额vs300:扣一次 round_trip(2×佣金+2×滑点+快照日印花税,PIT 分段见
   ashare_gauntlet.costs)的超额——无成本前向收益系统性虚高(五路文献研读结论);
   上界口径(持仓未平时实际只发生买入侧),参数与 scripts.factor_backtest 同默认同出处。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.pick_track
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from pathlib import Path

import pandas as pd

from ashare_gauntlet.costs import round_trip_cost_rate
from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.data.partition import assert_adj_complete, date_partition_files

CACHE = "data/cache"
OUT_DIR = "data/holdscore"

INDEX_CODE = "000300.SH"   # 沪深300 —— 配置基准(大盘宽基,主板宇宙的自然对照)
# regime 窗口 = 20 交易日 ≈ 一个自然月,复用代码库既有约定(pct20/ret20/reversal n=20)
REGIME_WINDOW = 20

# 成本参数(与 scripts.factor_backtest --commission/--slippage **同默认同出处**,两处口径
# 必须一致,否则回测与前向追踪的"成本后"不可比):
COMMISSION_RATE = 0.00025   # 单边佣金率:券商常见万2.5(用户合同参数,非库常数)
SLIPPAGE_RATE = 0.0015      # 单边滑点率:LWZ(2022 JFE)中国市场实测 15bp 取下沿


class EmptyIndexPullError(RuntimeError):
    """指数日线整段拉取返回 0 行 —— 沪深300 在非空交易区间不可能没有数据,
    空结果只能是拉取失败/权限问题;缓存它会毒化所有后续基准计算,fail-loud 不落盘。"""


class IncompleteIndexPullError(RuntimeError):
    """指数日线整段拉取中途丢行(实测:20260625 单日拉有、整段拉无 —— 即 fetch.py
    注释记录的镜像大响应掉行毛病)。指数在每个 A 股交易日都必有收盘,对照个股 daily
    缓存的交易日历校验;带洞序列会悄悄错移 regime 窗口/区间收益起点,fail-loud 不落盘,
    下次运行自动重拉。"""


def diff_picks(prev: list[str], curr: list[str]) -> dict[str, list[str]]:
    """两期名单 diff:new(本期新进)/ dropped(掉出)/ stay(留任)。保序。"""
    ps, cs = set(prev), set(curr)
    return {
        "new": [c for c in curr if c not in ps],
        "dropped": [p for p in prev if p not in cs],
        "stay": [c for c in curr if c in ps],
    }


def forward_returns(codes: list[str], snap_date: str, px: pd.DataFrame) -> dict[str, float]:
    """每只票 快照日(或其后首个交易日)→ 面板最新日 的前复权收益。缺价 NaN,不伪造。

    px 列:ts_code / trade_date / adj_close。用快照日之后首个可交易日为起点
    (快照可能落在周末/停牌),终点=该股面板内最新价——退市/长停股用其最后成交价,
    崩盘段收益计入(与回测修正版同一 survivorship 处理)。
    """
    out: dict[str, float] = {}
    for code in codes:
        g = px[(px["ts_code"] == code) & (px["trade_date"] >= snap_date)].sort_values("trade_date")
        ac = g["adj_close"].dropna()
        out[code] = float(ac.iloc[-1] / ac.iloc[0] - 1.0) if len(ac) >= 2 else (
            0.0 if len(ac) == 1 else math.nan)
    return out


def index_return(idx_df: pd.DataFrame, snap_date: str) -> float:
    """指数 快照日(或其后首个交易日)→ 表内最新日 的区间收益,与 forward_returns 同口径。

    idx_df 列:trade_date / close(指数点位,无需复权)。快照日非交易日 → 用其后首个
    交易日起算;快照日之后无数据(数据不足)→ NaN 不伪造;起点即最新日 → 0%。
    """
    c = idx_df[idx_df["trade_date"] >= snap_date].sort_values("trade_date")["close"].dropna()
    if len(c) == 0:
        return math.nan
    return float(c.iloc[-1] / c.iloc[0] - 1.0) if len(c) >= 2 else 0.0


def cost_adjusted_excess(port_ret: float, bench_ret: float, snap_date: str,
                         commission_rate: float, slippage_rate: float) -> float:
    """成本后超额 = 组合收益 − round_trip 成本率 − 基准收益(上界口径)。

    round_trip = 2×佣金 + 2×滑点 + 快照日印花税(PIT 分段,见 ashare_gauntlet.costs);
    把整段前向收益记**一次完整买卖**的成本——持仓未平时实际只发生买入侧,该列是保守
    上界而非实际成本。基准(沪深300)不扣成本:它是"不动的对照",扣了会自夸。
    收益 NaN(数据不足)→ 结果 NaN 传播,不伪造。
    """
    return port_ret - round_trip_cost_rate(snap_date, commission_rate, slippage_rate) - bench_ret


def regime_return(idx_df: pd.DataFrame, n: int) -> float:
    """regime 读数:最近 n 个交易日的指数涨跌 = close[-1]/close[-1-n] − 1。

    需要 n+1 个收盘价(n 个日收益的复合);行数不足 → NaN 不伪造。
    """
    c = idx_df.sort_values("trade_date")["close"].dropna()
    if len(c) < n + 1:
        return math.nan
    return float(c.iloc[-1] / c.iloc[-1 - n] - 1.0)


def load_index_daily(pro: object, ts_code: str, start_date: str, end_date: str,
                     cache_dir: str | Path = CACHE,
                     expected_days: list[str] | None = None) -> pd.DataFrame:
    """指数日线缓存:单文件 <cache_dir>/index_daily/<ts_code>.parquet,覆盖即用。

    已有缓存的 [min, max] 覆盖 [start, end] 且无洞 → 直接用(零 API 调用);不覆盖或
    有洞 → 整段重拉一次覆盖写(简单优先:单指数全区间也就千余行,一次调用拿完,避免
    每快照重复调 API;带洞旧缓存借此自愈)。expected_days=区间内应有的交易日(来自
    个股 daily 缓存日历),指数在每个 A 股交易日必有收盘:重拉后仍缺日 → fail-loud
    不落盘(见 IncompleteIndexPullError);空拉同理(见 EmptyIndexPullError)。
    """
    path = Path(cache_dir) / "index_daily" / f"{ts_code}.parquet"

    def _missing(df: pd.DataFrame) -> list[str]:
        if expected_days is None:
            return []
        have = set(str(d) for d in df["trade_date"])
        return [d for d in expected_days if d not in have]

    if path.exists():
        cached = pd.read_parquet(path)
        if not cached.empty and str(cached["trade_date"].min()) <= start_date \
                and str(cached["trade_date"].max()) >= end_date and not _missing(cached):
            return cached.sort_values("trade_date").reset_index(drop=True)
    df = call_with_retry(lambda: pro.index_daily(  # type: ignore[attr-defined]
        ts_code=ts_code, start_date=start_date, end_date=end_date))
    if df.empty:
        raise EmptyIndexPullError(
            f"index_daily {ts_code} 在 [{start_date}, {end_date}] 返回 0 行 —— "
            f"非空交易区间不可能无指数数据,疑为拉取失败/接口权限问题;拒绝缓存空值")
    holes = _missing(df)
    if holes:
        # 镜像大区间拉取会稳定漏个别日(实测 20260625:单日拉有、跨区间拉无)→
        # 缺日逐日补拉(单日路径可靠,同源真数据非伪造);补完仍缺才 fail-loud
        patches = [call_with_retry(lambda d=d: pro.index_daily(  # type: ignore[attr-defined]
            ts_code=ts_code, start_date=d, end_date=d)) for d in holes]
        df = pd.concat([df, *patches], ignore_index=True).drop_duplicates("trade_date")
        still = _missing(df)
        if still:
            raise IncompleteIndexPullError(
                f"index_daily {ts_code} 整段拉取 [{start_date}, {end_date}] 缺交易日 {holes},"
                f"单日补拉后仍缺 {still} —— 带洞序列会错移 regime 窗口/区间收益;"
                f"拒绝落盘,不把缺口当真值")
    df = df.sort_values("trade_date").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def main() -> None:
    snaps = sorted(glob.glob(f"{OUT_DIR}/*_factor.json"))
    if not snaps:
        raise SystemExit("无 factor 快照(先跑 scripts.factor_rank)")

    # 交易日历(日分区文件名即交易日)与快照日期,先于面板算出 —— 面板只需从最早快照日
    # 起读:forward_returns 只用 trade_date >= 快照日 的行,更早的 1000+ 个日文件读了
    # 也全被过滤掉(纯 IO 瘦身,口径零变化;全量读曾让单次运行 10 分钟起步)。
    # 走 date_partition_files:daily/ 实际混入过整段拉取文件,直接 glob 会把污染文件名
    # 当交易日、把重复行读进面板(见 data.partition 模块 docstring)
    day_files = date_partition_files(CACHE, "daily")
    trade_days = [os.path.basename(f)[:8] for f in day_files]
    snap_dates = [m.group(1) for p in snaps
                  if (m := re.match(r"(\d{8})_factor\.json", os.path.basename(p)))]
    first_snap = min(snap_dates) if snap_dates else trade_days[-1]

    # 价格面板(前复权),仅最早快照日及之后
    need = [f for f in day_files if os.path.basename(f)[:8] >= first_snap]
    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
                    for f in need], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in date_partition_files(CACHE, "adj_factor")
                    if os.path.basename(f)[:8] >= first_snap], ignore_index=True)
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    # fail-loud:adj_factor 缺日会让该日 adj_close=NaN 被 forward_returns 的 dropna 静默
    # 吞掉(前向收益起终点悄悄错位)—— 与 factor_rank 同一断言(data.partition)
    assert_adj_complete(px)
    px["adj_close"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    latest = str(px["trade_date"].max())

    def load_d10(path: str) -> list[str]:
        rows = json.load(open(path, encoding="utf-8"))
        return [r["ts_code"] for r in rows if r.get("decile") == 10]

    # 沪深300 日线(单文件缓存,一次拉够全区间:最早快照日 ∪ regime 窗口起点)
    regime_start = trade_days[-(REGIME_WINDOW + 1)] if len(trade_days) > REGIME_WINDOW else trade_days[0]
    idx_start = min(first_snap, regime_start)
    from ashare_gauntlet.data.env import load_env_local
    from ashare_gauntlet.data.tushare_source import make_pro_api
    load_env_local()
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    idx = load_index_daily(pro, INDEX_CODE, start_date=idx_start, end_date=latest,
                           expected_days=[d for d in trade_days if idx_start <= d <= latest])

    # ① 最近两期 diff(新进=动量陷阱高危,factcheck 优先核)
    if len(snaps) >= 2:
        prev_p, curr_p = snaps[-2], snaps[-1]
        d = diff_picks(load_d10(prev_p), load_d10(curr_p))
        nm = {r["ts_code"]: r.get("name", "") for r in json.load(open(curr_p, encoding="utf-8"))}
        nm.update({r["ts_code"]: r.get("name", "") for r in json.load(open(prev_p, encoding="utf-8"))})
        print(f"=== D10 变动:{os.path.basename(prev_p)} → {os.path.basename(curr_p)} ===")
        print(f"  新进 {len(d['new'])}(⚠️动量陷阱高危,factcheck 优先核): "
              + " ".join(f"{nm.get(c,'')}{c[-3:] and ''}({c[:6]})" for c in d["new"][:15]))
        print(f"  掉出 {len(d['dropped'])}: " + " ".join(f"{nm.get(c,'')}({c[:6]})" for c in d["dropped"][:15]))
        print(f"  留任 {len(d['stay'])}")

    # ② 各历史快照 D10 的前向收益 vs 双基准(宇宙内等权=选股能力,沪深300=配置效果)
    print(f"\n=== D10 前向收益 vs 双基准(至 {latest};样本外命中率证据,逐期积累)===")
    print(f"{'快照日':>10}{'D10只数':>7}{'D10均收益':>10}{'基准均收益':>10}{'超额':>8}"
          f"{'沪深300收益':>10}{'超额vs300':>9}{'成本后超额vs300':>12}")
    for p in snaps:
        m = re.match(r"(\d{8})_factor\.json", os.path.basename(p))
        if not m:
            continue
        snap = m.group(1)
        if snap >= latest:
            continue
        rows = json.load(open(p, encoding="utf-8"))
        d10 = [r["ts_code"] for r in rows if r.get("decile") == 10]
        uni = [r["ts_code"] for r in rows]
        fr = forward_returns(d10, snap, px)
        fu = forward_returns(uni, snap, px)
        d10_m = pd.Series(fr).mean()
        uni_m = pd.Series(fu).mean()
        hs = index_return(idx, snap)
        net300 = cost_adjusted_excess(d10_m, hs, snap, COMMISSION_RATE, SLIPPAGE_RATE)
        print(f"{snap:>10}{len(d10):>7}{d10_m*100:>+9.1f}%{uni_m*100:>+9.1f}%{(d10_m-uni_m)*100:>+7.1f}%"
              f"{hs*100:>+11.1f}%{(d10_m-hs)*100:>+10.1f}%{net300*100:>+15.1f}%")
    print("(超额>0=跑赢自身宇宙=选股能力;超额vs300>0=跑赢配置基准;样本少时噪声大,别过度解读单期)")
    print(f"(成本后超额vs300:扣一次 round_trip=2×佣金万{COMMISSION_RATE*10000:g}+2×滑点"
          f"{SLIPPAGE_RATE*10000:g}bp+快照日印花税——上界口径,持仓未平实际成本更低;"
          f"与 factor_backtest 同默认同出处)")

    # regime 读数:D10 跑输时先分清是 α(选股)还是 β(市场)的问题
    rg = regime_return(idx, REGIME_WINDOW)
    print(f"\nregime:最近{REGIME_WINDOW}交易日 沪深300 {rg*100:+.1f}%"
          f"(D10 与 300 同跌=β 问题,300 涨而 D10 跌=α 问题)")


if __name__ == "__main__":
    main()
