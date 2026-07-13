"""composite 端到端组合回测 —— 回答外部评审二轮:"单因子各自过门 ≠ 生产 D10 组合被证实"。

复刻生产链路(合成/分位/定档与 scripts.factor_rank **同一函数**,不另造口径):
EP/BP/IVOL 原值 → factor_percentile(行业+市值双中性→百分位,IVOL 负向)→ composite
等权 → to_decile;组合 T+1 开盘等权买入、持有 --fwd 日、入场一字涨停剔除、退出侧
一字跌停/停牌顺延(与 scripts.factor_backtest 同引擎组件)。

一次跑出八个口径:
- D1..D10   :无 tier 预过滤宇宙(与单因子五门证据同宇宙,验单调性)
- PROD      :生产宇宙(lean_tier ∈ 🟢🟡 + 有披露财务 + 剔ST,同 factor_rank)→ D10
- PROD_G    :PROD 的 D10 ∩ 🟢
- PROD_GX   :PROD_G 再剔 data/profile.json 行业(反事实:当前 profile 应用于全史)
- PROD_DEDT :EP → 扣非 TTM 口径(全覆盖版,R4)
- CS_EP/CS_EPD:common-support 同池对照(每期限定四因子全可得的同一批票,R6——
  两行之差=纯 EP 定义效应,剥离覆盖池与构件差异)
- PROD_XP   :PROD 剔除污染标记(pe>0 且扣非TTM≤0=主业亏损靠非经常盈利)后的边际

生产复刻经三轮审查对齐(2026-07-13):①先滤后排(先排后滤实测每期 D10 与生产
对称差 ~9%);②打分在 t 日全宇宙完成,T+1 一字涨停只从**已选成员**剔除、不重排名
(P0-1:先剔后排=用 T+1 信息改写分位边界);③宇宙名录=stock_basic L+D+P 全状态
(P0-2:仅 L 会把后来退市的股票整段剔除,幸存者偏差回门);④扣非 TTM 构件按期
严格 PIT 现算(dedt_ttm_pit,R6,废弃"更正后终值"行级近似)。残余差异(报告尾
打印):ST=最终名称近似(namechange PIT 待补)。
成本:round_trip(佣金+滑点+卖出日印花税 PIT 分段)× 组合真实换手 τ(首期建仓 τ=1)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.composite_backtest
       [--fwd 21] [--start YYYYMMDD] [--commission 0.00025] [--slippage 0.0015]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ashare_gauntlet.backtest import information_coefficient, newey_west_tstat
from ashare_gauntlet.config import CACHE_DIR as CACHE, HOLDSCORE_DIR, tushare_pro
from ashare_gauntlet.costs import round_trip_cost_rate
from ashare_gauntlet.data.fetch import call_with_retry, fetch_market_day
from ashare_gauntlet.data.partition import assert_adj_complete, date_partition_files
from ashare_gauntlet.factor_model import (
    composite,
    daily_returns,
    factor_percentile,
    ivol_capm,
    to_decile,
)
from ashare_gauntlet.record import lean_tier
from ashare_gauntlet.screen import board_of
from scripts.aggressive_pick import load_excluded_industries
from scripts.factor_backtest import (
    MAIN,
    _load,
    _pit,
    defer_note,
    first_sellable_open,
    leg_turnover,
    one_word_limit_down,
    one_word_limit_up,
)

MOM_LB = 250   # 与 factor_backtest 同锚:保证换仓日集合一致(N 可比),非本实验用到动量


def max_drawdown(returns: pd.Series) -> float:
    """期收益序列 → 复利净值最大回撤(≤0)。NaN 期跳过;空序列 NaN(无数据≠零回撤)。"""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def industry_hhi(industries: pd.Series) -> float:
    """等权组合行业集中度(Herfindahl):Σ(行业权重²)。1=全一个行业;空组合 NaN。"""
    if industries.empty:
        return float("nan")
    w = industries.value_counts(normalize=True)
    return float((w * w).sum())


def top_contrib_share(contrib: pd.Series) -> float:
    """单票贡献集中度:|最大贡献| / Σ|贡献|(gauntlet 单票伪装守卫同语义)。空则 NaN。"""
    c = contrib.dropna()
    gross = float(c.abs().sum())
    if not gross:
        return float("nan")
    return float(c.abs().max() / gross)


def dedt_ttm_pit(fd: pd.DataFrame, asof: str) -> pd.Series:
    """严格 PIT 扣非 TTM(评审三轮 R6):每个构件只用 ann_date<=asof 时已知的最新版本。

    输入=全市场 fina 行(ts_code/end_date/ann_date/profit_dedt,YTD 累计,已按
    ts_code/end_date/ann_date 排序)。TTM = 最新已披露期 YTD + 上年年报 − 上年同期
    YTD;最新期为年报(1231)时 TTM=自身;任一构件在视界内缺失 → NaN(fail-honest);
    一切未披露的股票缺席结果。取代旧行级近似(构件用最终更正值=微量后验,评审实测
    复现后废弃)。
    """
    vis = fd[fd["ann_date"] <= asof]
    if vis.empty:
        return pd.Series(dtype=float)
    # 视界内每 (股, 期) 的最新已知版本(输入行序已按 ann_date 升序,last=最新公告)
    latest = vis.groupby(["ts_code", "end_date"])["profit_dedt"].last()
    cur = vis.groupby("ts_code", sort=False).tail(1)   # 每股最新已披露期(行序契约)
    out: dict[str, float] = {}
    for c, e, v in zip(cur["ts_code"], cur["end_date"], cur["profit_dedt"]):
        e = str(e)
        if pd.isna(v):
            out[str(c)] = float("nan")
            continue
        if e.endswith("1231"):
            out[str(c)] = float(v)  # type: ignore[arg-type]
            continue
        pa = latest.get((c, str(int(e[:4]) - 1) + "1231"))
        ps = latest.get((c, str(int(e[:4]) - 1) + e[4:]))
        if pa is None or ps is None or pd.isna(pa) or pd.isna(ps):
            out[str(c)] = float("nan")
        else:
            out[str(c)] = float(v) + float(pa) - float(ps)  # type: ignore[arg-type]
    return pd.Series(out)


PORTS = ("D10", "PROD", "PROD_G", "PROD_GX", "PROD_DEDT", "CS_EP", "CS_EPD", "PROD_XP")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, default=21)
    ap.add_argument("--start", default=None)
    ap.add_argument("--commission", type=float, default=0.00025,
                    help="单边佣金率(万2.5,与 factor_backtest 同出处)")
    ap.add_argument("--slippage", type=float, default=0.0015,
                    help="单边滑点率(LWZ 2022 JFE 中国实测 15bp 下沿)")
    a = ap.parse_args(argv)

    excluded = load_excluded_industries()
    fina = _load("fina_indicator", ["roe", "netprofit_yoy", "dt_netprofit_yoy", "tr_yoy", "ocfps"])
    # R4 扣非对照:profit_dedt(YTD 累计)→ 逐行 TTM(EP 污染实验,methodology §7-⑤)
    print("加载 profit_dedt 全市场行(扣非 TTM 构件按期严格 PIT 现算)…", flush=True)
    fdedt = _load("fina_indicator", ["profit_dedt"])
    # 脏 ann_date(NaN→astype(str)→"nan" 会恒排最后赢得构件查表)从严剔除,
    # 与生产 latest_rows 的三键排序防线同精神(审查 P2-7)
    fdedt = fdedt[fdedt["ann_date"].str.fullmatch(r"\d{8}", na=False)]
    pro = tushare_pro()
    # 宇宙名录 = L+D+P 全状态(评审三轮 P0-2:仅 list_status="L" 会把后来退市的
    # 股票从宇宙 B 整段历史剔除——退市财务回填白做,幸存者偏差从后门回来)
    frames = [call_with_retry(lambda s=s: pro.stock_basic(
        list_status=s, fields="ts_code,industry,name")) for s in ("L", "D", "P")]
    sb = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code").set_index("ts_code")
    ind_all = sb["industry"].fillna("其他")
    # ST 剔除=最终名称近似(历史逐日 ST 状态需 namechange PIT 面板,记为残余偏差:
    # 后来戴帽的被错误早剔、当时戴帽后摘帽的被错误纳入;规模已在报告尾量化)
    non_st = ~sb["name"].astype(str).str.contains("ST", na=False)

    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"])
                    for f in date_partition_files(CACHE, "daily")], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in date_partition_files(CACHE, "adj_factor")], ignore_index=True)
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    assert_adj_complete(px)
    px["aclose"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    px["aopen"] = px["open"].astype(float) * px["adj_factor"].astype(float)
    dates = sorted(px["trade_date"].unique())
    close_p = px.pivot_table(index="trade_date", columns="ts_code", values="aclose")
    open_p = px.pivot_table(index="trade_date", columns="ts_code", values="aopen")
    ret_p = daily_returns(close_p)          # 停牌 NaN 保持(伪低波防线,与引擎同)
    mkt = ret_p.mean(axis=1)

    di = {d: i for i, d in enumerate(dates)}
    month_last: dict[str, str] = {}
    for d in dates:
        month_last[d[:6]] = d
    rebal = [d for d in month_last.values() if di[d] >= MOM_LB and di[d] + 1 + a.fwd < len(dates)]
    if a.start:
        rebal = [d for d in rebal if d >= a.start]
    print(f"加载完成:{len(dates)}交易日 → {len(rebal)}个月度换仓日;宇宙=主板,"
          f"构造与 factor_rank 同函数(factor_percentile/composite/to_decile)", flush=True)

    rows: list[dict] = []
    prev_sets: dict[str, set | None] = {p: None for p in PORTS}
    contrib: dict[str, dict[str, float]] = {p: {} for p in PORTS}
    for k, t in enumerate(rebal):
        it = di[t]
        codes = [str(c) for c in close_p.columns[close_p.loc[t].notna()] if board_of(str(c)) in MAIN]
        if len(codes) < 50:
            continue
        idx = pd.Index(codes)
        entry_date = dates[it + 1]
        # T+1 一字涨停 = 可执行性约束,只作用于**已选组合成员**,不得进入打分宇宙
        # (评审三轮 P0-1:生产在 t 日不知明天涨停名单;先剔后排=用 T+1 信息改写
        # 行业中位/市值桶/分位边界,属前视)。锁定票同样剔出基准(买不进的收益不可得)
        locked = one_word_limit_up(fetch_market_day(pro, "daily", entry_date, CACHE),
                                   fetch_market_day(pro, "stk_limit", entry_date, CACHE), codes)
        locked_idx = pd.Index([c for c in locked])
        # 前向收益(与引擎同):T+1 开盘 → 窗口末,退出一字跌停/停牌顺延到首个可卖开盘
        win = open_p.iloc[it + 1: it + 2 + a.fwd][codes]
        entry = win.iloc[0]
        exit_ = win.ffill().iloc[-1].copy()
        exit_pos = it + 1 + a.fwd
        _ld_cache: dict[int, set[str]] = {}

        def _locked(pos: int, c: str, _pool: list[str] = codes) -> bool:
            if pos not in _ld_cache:
                d_ = dates[pos]
                _ld_cache[pos] = one_word_limit_down(
                    fetch_market_day(pro, "daily", d_, CACHE),
                    fetch_market_day(pro, "stk_limit", d_, CACHE), _pool)
            return c in _ld_cache[pos]

        locked_exit = one_word_limit_down(fetch_market_day(pro, "daily", dates[exit_pos], CACHE),
                                          fetch_market_day(pro, "stk_limit", dates[exit_pos], CACHE), codes)
        suspended = {c for c in codes if pd.isna(open_p.iloc[exit_pos].get(c)) and pd.notna(entry.get(c))}
        n_def = n_unres = def_days = 0
        for c in locked_exit | suspended:
            r = first_sellable_open(open_p[c], exit_pos + 1, lambda j, _c=c: _locked(j, _c))
            if r is None:
                n_unres += 1
            else:
                exit_[c] = r[0]
                n_def += 1
                def_days += r[1] + 1
        fwd = exit_ / entry - 1.0

        # 因子原值(与 factor_rank 同口径:daily_basic 快照 + IVOL 21日 CAPM 残差)
        db = fetch_market_day(pro, "daily_basic", t, CACHE).set_index("ts_code")
        mv = pd.to_numeric(db["total_mv"], errors="coerce").reindex(idx)
        pe = pd.to_numeric(db["pe_ttm"], errors="coerce").reindex(idx)
        pb_ = pd.to_numeric(db["pb"], errors="coerce").reindex(idx)
        raw = pd.DataFrame(index=idx)
        raw["EP"] = (1.0 / pe).where(pe > 0)
        raw["BP"] = (1.0 / pb_).where(pb_ > 0)
        raw["IVOL"] = ivol_capm(ret_p.iloc[: it + 1], mkt.iloc[: it + 1], 21).reindex(idx)
        # 扣非 EP(R4/R6):严格 PIT TTM(元)/ 总市值(daily_basic total_mv 单位=万元);
        # 与 EP 同样仅在 E>0 下有定义(价值因子语义)
        dedt = dedt_ttm_pit(fdedt, t).reindex(idx)
        raw["EPD"] = (dedt / (mv * 1e4)).where((mv > 0) & (dedt > 0))
        # 污染探针(R6,定义性零阈值):主业不赚钱(扣非 TTM≤0)而市场按含非经常
        # 口径给出正 PE——好想你式"一次性收益抬 EP"的机器可判形态;dedt 未知不标
        poll_mark = (pe > 0) & dedt.notna() & (dedt <= 0)
        ind = ind_all.reindex(idx).fillna("其他")
        logmv = np.log(mv.where(mv > 0))
        pf = _pit(fina, t)
        tier = pd.Series(
            [lean_tier(pf["netprofit_yoy"].get(c), pf["dt_netprofit_yoy"].get(c),
                       pf["tr_yoy"].get(c), ocfps=pf["ocfps"].get(c), roe=pf["roe"].get(c))
             for c in idx], index=idx)

        def score_deciles(sub: pd.Index, ep_col: str = "EP") -> "pd.Series | None":
            """给定宇宙内按生产构造打分定档。**先滤后排**(对抗审查 P1-1:生产是
            tier→剔ST→pe>0→三因子全齐先过滤、再在滤后池算百分位;先排后滤会让
            pe≤0 等出局票污染 BP/IVOL 的百分位池/行业中位/size 桶界)。

            ep_col="EPD" 时为 R4 对照:EP 换扣非 TTM 口径,其余构造不变。
            """
            ok = raw.loc[sub, [ep_col, "BP", "IVOL"]].notna().all(axis=1)
            pool = sub[ok]                      # 入池门槛先行(composite_inputs_complete 同序)
            if len(pool) < 50:
                return None
            f = pd.DataFrame(index=pool)
            f["f_EP"] = factor_percentile(raw.loc[pool, ep_col], ind[pool], True, logmv=logmv[pool])
            f["f_BP"] = factor_percentile(raw.loc[pool, "BP"], ind[pool], True, logmv=logmv[pool])
            f["f_IVOL"] = factor_percentile(raw.loc[pool, "IVOL"], ind[pool], False, logmv=logmv[pool])
            return to_decile(composite(f))

        members: dict[str, pd.Index] = {}
        dec_a = score_deciles(idx)              # 宇宙A:无 tier 预过滤(单调性/与单因子可比)
        if dec_a is None:
            continue
        # 打分完成后才应用 T+1 可执行性:锁定票从**已选成员**与基准中剔除,排名不重算
        # (P0-1 修复;组合语义=想买但买不进,只能少持这一只)
        members["D10"] = dec_a.index[dec_a == 10].difference(locked_idx)
        exec_idx = idx.difference(locked_idx)
        row: dict = {"date": t, "mkt_fwd": float(fwd[exec_idx].mean()),
                     "cost_rt": round_trip_cost_rate(entry_date, a.commission, a.slippage,
                                                     sell_date=dates[exit_pos]),
                     "exit_deferred": n_def, "exit_unresolved": n_unres,
                     "n_locked": len(locked)}
        for q in range(1, 11):
            m = dec_a.index[dec_a == q].difference(locked_idx)
            row[f"ret_Q{q}"] = float(fwd[m].mean()) if len(m) else float("nan")

        # 宇宙B:生产复刻(与 factor_rank 同序:tier 🟢🟡 + 有已披露财务 + 剔 ST[最终名称近似])。
        # 对抗审查 P1-1 实测:漏掉 has_fina/ST 时每期 D10 与生产对称差 ~9%(含 ST 票混入)
        has_fina = idx.isin(pf.index)
        sub_b = idx[tier.isin(["🟢", "🟡"]) & has_fina & non_st.reindex(idx).fillna(False)]
        dec_b = score_deciles(sub_b)
        if dec_b is not None:
            d10b = dec_b.index[dec_b == 10]
            row["locked_in_prod"] = int(len(d10b.intersection(locked_idx)))   # 想买没买进的
            d10b = d10b.difference(locked_idx)
            members["PROD"] = d10b
            members["PROD_G"] = pd.Index([c for c in d10b if tier[c] == "🟢"])
            members["PROD_GX"] = pd.Index([c for c in members["PROD_G"] if ind[c] not in excluded])
            row["hhi_PROD"] = industry_hhi(ind[d10b])
            # R6 污染探针落库:标记股入 D10 频率 + 标记组当期收益 + 剔除后的边际组合
            poll_in = pd.Index([c for c in d10b if bool(poll_mark.get(c, False))])
            row["n_poll_prod"] = int(len(poll_in))
            row["ret_poll_prod"] = float(fwd[poll_in].mean()) if len(poll_in) else float("nan")
            members["PROD_XP"] = d10b.difference(poll_in)
        row["n_poll_universe"] = int(poll_mark[sub_b].sum())
        # R4:同宇宙、EP→扣非 TTM 口径的对照组合 + 因子级 IC 对照(可执行宇宙百分位)
        dec_d = score_deciles(sub_b, ep_col="EPD")
        if dec_d is not None:
            members["PROD_DEDT"] = dec_d.index[dec_d == 10].difference(locked_idx)
            if dec_b is not None and len(members["PROD"]):
                row["ovl_dedt"] = len(set(members["PROD"]) & set(members["PROD_DEDT"])) / len(members["PROD"])
        # R6 common-support(评审三轮):每期限定四因子全可得的**同一批票**,分别用
        # EP/扣非EP 排名——成员池完全一致,收益/回撤差异才可归因于 EP 定义本身
        cs = sub_b[raw.loc[sub_b, ["EP", "EPD", "BP", "IVOL"]].notna().all(axis=1)]
        dec_ce = score_deciles(cs)
        dec_cd = score_deciles(cs, ep_col="EPD")
        if dec_ce is not None and dec_cd is not None:
            members["CS_EP"] = dec_ce.index[dec_ce == 10].difference(locked_idx)
            members["CS_EPD"] = dec_cd.index[dec_cd == 10].difference(locked_idx)
            if len(members["CS_EP"]):
                row["ovl_cs"] = len(set(members["CS_EP"]) & set(members["CS_EPD"])) / len(members["CS_EP"])
        row["IC_EP"] = information_coefficient(
            factor_percentile(raw["EP"], ind, True, logmv=logmv)[exec_idx], fwd[exec_idx])
        row["IC_EPD"] = information_coefficient(
            factor_percentile(raw["EPD"], ind, True, logmv=logmv)[exec_idx], fwd[exec_idx])
        for p in PORTS:
            m = members.get(p)
            if m is None or not len(m):
                row[f"ret_{p}"] = float("nan")
                row[f"n_{p}"] = 0
                row[f"TO_{p}"] = float("nan")
                continue
            row[f"ret_{p}"] = float(fwd[m].mean())
            row[f"n_{p}"] = int(len(m))
            # 首期建仓成本计 τ=1(组合口径,区别于引擎逐因子腿的首期 NaN 剔除——
            # 组合的建仓是真实要付的一次性成本,不计=毛/净样本数错位,审查 P2-1)
            tau = leg_turnover(prev_sets[p], set(m))
            row[f"TO_{p}"] = 1.0 if prev_sets[p] is None else tau
            prev_sets[p] = set(m)
            # 单票贡献:分母=有效成交成员数(与组合收益的 skipna 均值同分母,审查 P2-3)
            valid = [c for c in m if pd.notna(fwd.get(c))]
            for c in valid:
                contrib[p][c] = contrib[p].get(c, 0.0) + float(fwd[c]) / len(valid)
        print(f"  {k + 1}/{len(rebal)} {t} 剔涨停{len(locked)} n(D10/PROD/G/GX)="
              f"{row.get('n_D10', 0)}/{row.get('n_PROD', 0)}/{row.get('n_PROD_G', 0)}/"
              f"{row.get('n_PROD_GX', 0)}{defer_note(n_def, def_days, n_unres)}", flush=True)
        rows.append(row)

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("无有效换仓期(检查 --start / 缓存覆盖)")
    # 先落盘再报告:149 期逐日计算约数十分钟,报告层的任何 bug 不允许毁掉计算成果
    os.makedirs(HOLDSCORE_DIR, exist_ok=True)
    res.to_json(f"{HOLDSCORE_DIR}/composite_backtest.json", orient="records", force_ascii=False, indent=2)
    y = res["date"].str[:4]

    print(f"\n=== composite 端到端组合回测(N={len(res)},{res['date'].min()}→{res['date'].max()},"
          f"持有{a.fwd}日,T+1开盘,一字板双侧约束,成本=τ×round_trip)===")
    dec_means = [(q, res[f"ret_Q{q}"].mean() * 100) for q in range(1, 11)]
    # Spearman = 秩的 Pearson(pandas 3.x 的 method="spearman" 需要 scipy,这里手工取秩零新依赖)
    mono = pd.Series([m for _, m in dec_means]).rank().corr(pd.Series(range(1, 11)).rank())
    print("D1→D10 期均收益%(宇宙A,无tier过滤):"
          + " ".join(f"{m:+.2f}" for _, m in dec_means) + f" | 单调性ρ={mono:+.2f}")
    print(f"{'组合':>8}{'期数':>5}{'均只数':>7}{'超额毛%':>8}{'NW t':>7}{'换手':>6}{'超额净%':>8}"
          f"{'年胜率':>7}{'涨市%':>7}{'跌市%':>7}{'绝对MaxDD':>10}{'单票贡献':>9}")
    for p in PORTS:
        ex = res[f"ret_{p}"] - res["mkt_fwd"]
        net = ex - res[f"TO_{p}"] * res["cost_rt"]
        _, tnw, _ = newey_west_tstat(ex.dropna())
        yr = net.groupby(y).mean().dropna()     # 全 NaN 年不计入胜率分母(审查 P2-4)
        up = ex[res["mkt_fwd"] > 0].mean() * 100
        dn = ex[res["mkt_fwd"] <= 0].mean() * 100
        absnet = res[f"ret_{p}"] - res[f"TO_{p}"] * res["cost_rt"]
        print(f"{p:>8}{ex.notna().sum():>5}{res[f'n_{p}'].replace(0, np.nan).mean():>7.0f}"
              f"{ex.mean() * 100:>+7.2f}%{tnw:>+7.2f}{res[f'TO_{p}'].mean():>6.0%}"
              f"{net.mean() * 100:>+7.2f}%{(yr > 0).mean() * 100:>6.0f}%{up:>+6.2f}%{dn:>+6.2f}%"
              f"{max_drawdown(absnet):>10.1%}{top_contrib_share(pd.Series(contrib[p])):>9.1%}")
    print(f"PROD 行业集中度 HHI 均值:{res['hhi_PROD'].mean():.3f}"
          f"(等权下限 E[1/n]={(1 / res['n_PROD'].replace(0, np.nan)).mean():.3f})")
    # R4 因子级对照:EP(含非经常) vs EPD(扣非 TTM)
    for nm in ("IC_EP", "IC_EPD"):
        ic = res[nm].dropna()
        _, tnw, _ = newey_west_tstat(ic)
        print(f"{nm}: IC均值{ic.mean():+.3f} | NW t{tnw:+.2f} | N={len(ic)}")
    if "ovl_dedt" in res:
        print(f"D10(EP) 与 D10(扣非EP) 平均重合率:{res['ovl_dedt'].mean():.0%}"
              f"(平均每期换血 {(1 - res['ovl_dedt'].mean()) * res['n_PROD'].mean():.0f} 只)")
    if "ovl_cs" in res:
        print(f"common-support 同宇宙对照(R6,成员池完全一致):D10(EP) vs D10(扣非EP) "
              f"重合率 {res['ovl_cs'].mean():.0%}——CS_EP/CS_EPD 两行之差即纯 EP 定义效应")
    # R6 污染探针(定义性标记:pe>0 且扣非 TTM≤0 = 主业亏损靠非经常盈利)
    print(f"污染探针:宇宙B均 {res['n_poll_universe'].mean():.0f} 只/期被标记;"
          f"落入 PROD D10 均 {res['n_poll_prod'].mean():.2f} 只/期"
          f"(占 D10 成员 {res['n_poll_prod'].sum() / max(res['n_PROD'].sum(), 1):.2%});"
          f"标记组期均收益 {res['ret_poll_prod'].mean() * 100:+.2f}% vs PROD 整体 "
          f"{res['ret_PROD'].mean() * 100:+.2f}%;剔除标记的边际影响=上表 PROD_XP 行")
    yr_tbl = (res["ret_PROD"] - res["mkt_fwd"] - res["TO_PROD"] * res["cost_rt"]).groupby(y)
    print("PROD 逐年净超额%(期均):" + " ".join(
        f"{yy}:{v.mean() * 100:+.2f}" for yy, v in yr_tbl))
    print(f"退出顺延合计 {res['exit_deferred'].sum()} 次,未解(退市终局){res['exit_unresolved'].sum()} 次;"
          f"T+1 一字涨停均锁 {res['n_locked'].mean():.1f} 只/期,其中锁在 PROD 内(想买没买进)"
          f"均 {res['locked_in_prod'].mean():.2f} 只/期")
    print(f"已知与生产的残余差异(评审三轮后口径):①宇宙名录=L+D+P 全状态(退市股在池),"
          f"但 ST=最终名称近似(历史逐日戴帽状态需 namechange PIT——当前按名称剔除 "
          f"{int((~non_st).sum())} 只,双向偏差:后来戴帽的被错误早剔/摘帽的被错误纳入);"
          f"②退出顺延跨税改日时印花税取计划退出日段(量级趋零);③⚡🎰/dma20 展示列不入分。"
          f"(dedt TTM 构件已改严格 PIT 现算,旧'更正后终值'近似废弃——R6)")
    print("→ 明细 data/holdscore/composite_backtest.json(已在报告前落盘)")


if __name__ == "__main__":
    main()
