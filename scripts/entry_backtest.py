"""入场规则独立实验(spec §6.2/§6.3)——在已审计 PROD 成员上评测,过门才可产 BUY。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.entry_backtest

实验登记(防事后挑选,写死于此):
- 候选=data/holdscore/composite_members.json(composite_backtest 落盘的逐期 PROD 成员);
- 受测规则与先验见 ashare_gauntlet/entry_model.py 头注;
- **预先指定显著性主窗 = 21 交易日**;5/10/42 仅参考(42 与月距明确重叠,主窗在
  20 交易日月份至多轻微重叠——由 NW HAC 吸收,报告 ACF1 供核);
- R_GAP 语义=限价单:开盘前挂"≤前收×(1+阈值)"限价,开盘价同时是成交条件与成交价
  (Codex review 确认该建模无必然前视;若解释为看到开盘价再发市价单则含零延迟假设)。

口径(Codex review 两 P0 修复后):
- 成本显式建模:两臂各自逐期换手 τ(leg_turnover,首期建仓 τ=1)× round_trip
  (佣金万2.5+滑点15bp+卖出日印花税 PIT 分段,出处同 factor_backtest);
  net_diff = (规则臂毛-τ_r·cost) − (基础臂毛-τ_b·cost)——"每笔成本相同故互消"的
  旧论证已废弃(两臂跨期留任率不同,换手不互消);
- 退出顺延:全部四窗启用一字跌停/停牌顺延(one_word_limit_down+first_sellable_open,
  与主引擎同组件);
- 市场切片=逐股 20 日收益的横截面等权(旧"平均价格涨跌"口径废弃——高价股权重
  失真);LOYO=真剔年(剔除该年后在剩余全样本重估符号)。
- common support=四特征齐备∩入场可成交的统一支持集(每期损失如实打印;各规则
  分别支持集留待需要时扩展)。
"""
from __future__ import annotations

import json
import os

import pandas as pd

from ashare_gauntlet.backtest import newey_west_tstat
from ashare_gauntlet.config import CACHE_DIR as CACHE, HOLDSCORE_DIR
from ashare_gauntlet.costs import round_trip_cost_rate
from ashare_gauntlet.data.fetch import fetch_market_day
from ashare_gauntlet.data.partition import assert_adj_complete, date_partition_files
from ashare_gauntlet.config import tushare_pro
from ashare_gauntlet.entry_model import dma20, gap_pct, gate_verdict, ret_n
from ashare_gauntlet.execution import entry_readiness
from scripts.factor_backtest import first_sellable_open, leg_turnover, one_word_limit_down

HORIZONS = (5, 10, 21, 42)
MAIN_H = 21                      # 预先指定主窗(登记)
COMMISSION, SLIPPAGE = 0.00025, 0.0015   # 出处同 factor_backtest 默认(万2.5/LWZ 15bp)
RULES = {                        # 规则 → (主阈值, 邻域阈值列表, 通过条件)
    "R_READY": (None, [None], "label==右侧✓"),
    "R_DMA20": (0.0, [-0.02, 0.0, 0.02], "dma20<=thr"),
    "R_RET5": (0.0, [-0.02, 0.0, 0.02], "ret5<=thr"),
    "R_GAP": (0.03, [0.02, 0.03, 0.04], "gap<=thr(限价单语义)"),
}


def _rule_mask(rule: str, thr, feats: pd.DataFrame) -> pd.Series:
    if rule == "R_READY":
        return feats["ready"] == "右侧✓"
    if rule == "R_DMA20":
        return feats["dma20"] <= thr
    if rule == "R_RET5":
        return feats["ret5"] <= thr
    if rule == "R_GAP":
        return feats["gap"] <= thr
    raise ValueError(rule)


def _key(rule: str, thr) -> str:
    return f"{rule}@{thr}" if thr is not None else rule


def main() -> None:
    members = json.load(open(f"{HOLDSCORE_DIR}/composite_members.json", encoding="utf-8"))
    if not members:
        raise SystemExit("composite_members.json 为空——先跑 scripts.composite_backtest")
    pro = tushare_pro()

    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close", "vol"])
                    for f in date_partition_files(CACHE, "daily")], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in date_partition_files(CACHE, "adj_factor")], ignore_index=True)
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    assert_adj_complete(px)
    px["aclose"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    px["aopen"] = px["open"].astype(float) * px["adj_factor"].astype(float)
    close_p = px.pivot_table(index="trade_date", columns="ts_code", values="aclose")
    open_p = px.pivot_table(index="trade_date", columns="ts_code", values="aopen")
    vol_p = px.pivot_table(index="trade_date", columns="ts_code", values="vol")
    dates = list(close_p.index)
    di = {d: i for i, d in enumerate(dates)}
    # 市场切片:逐股 20 日收益 → 横截面等权(Codex P1 修复:平均价格口径高价股失真)
    univ20 = close_p.pct_change(20).mean(axis=1)

    per_period: list[dict] = []
    prev_sets: dict[str, "set | None"] = {}
    support_loss = 0
    for m in members:
        t = m["date"]
        it = di[t]
        if it + 1 + max(HORIZONS) >= len(dates):
            continue
        cols = [c for c in m["prod"] if c in close_p.columns]
        hist = close_p.iloc[: it + 1][cols]
        entry = open_p.iloc[it + 1][cols]
        feats = pd.DataFrame(index=cols)
        feats["dma20"] = dma20(hist)
        feats["ret5"] = ret_n(hist, 5)
        feats["gap"] = gap_pct(entry, close_p.iloc[it][cols])
        ready = {}
        for c in cols:
            try:
                ready[c] = entry_readiness(hist[c], vol_p.iloc[: it + 1][c])["label"]
            except (ValueError, KeyError):
                ready[c] = None                          # 历史不足:common support 外
        feats["ready"] = pd.Series(ready)
        cs = feats.dropna().index.intersection(entry.dropna().index)   # 统一支持集
        support_loss += len(cols) - len(cs)
        if len(cs) < 10:
            continue

        # 逐窗退出(一字跌停/停牌顺延,与主引擎同组件——Codex P0-2 修复)
        _ld_cache: dict[int, set[str]] = {}
        pool = list(cs)

        def _locked(pos: int, c: str) -> bool:
            if pos not in _ld_cache:
                d_ = dates[pos]
                _ld_cache[pos] = one_word_limit_down(
                    fetch_market_day(pro, "daily", d_, CACHE),
                    fetch_market_day(pro, "stk_limit", d_, CACHE), pool)
            return c in _ld_cache[pos]

        row: dict = {"date": t, "year": t[:4],
                     "regime_up": bool(univ20.iloc[it] > 0) if pd.notna(univ20.iloc[it]) else None,
                     "n_base": len(cs)}
        fwd_by_h: dict[int, pd.Series] = {}
        for h in HORIZONS:
            exit_pos = it + 1 + h
            ex = open_p.iloc[it + 1: exit_pos + 1][cs].ffill().iloc[-1].copy()
            locked_exit = _ld_cache.get(exit_pos)
            if locked_exit is None:
                locked_exit = one_word_limit_down(
                    fetch_market_day(pro, "daily", dates[exit_pos], CACHE),
                    fetch_market_day(pro, "stk_limit", dates[exit_pos], CACHE), pool)
                _ld_cache[exit_pos] = locked_exit
            susp = {c for c in cs if pd.isna(open_p.iloc[exit_pos].get(c)) and pd.notna(entry.get(c))}
            for c in (locked_exit & set(cs)) | susp:
                r = first_sellable_open(open_p[c], exit_pos + 1, lambda j, _c=c: _locked(j, _c))
                if r is not None:
                    ex[c] = r[0]
            fwd_by_h[h] = ex / entry[cs] - 1.0
            row[f"cost_{h}"] = round_trip_cost_rate(dates[it + 1], COMMISSION, SLIPPAGE,
                                                    sell_date=dates[exit_pos])
            row[f"base_{h}"] = float(fwd_by_h[h].mean())

        # 两臂换手(成员集合口径,首期建仓 τ=1;窗间共享成员故 τ 与 h 无关)
        base_set = set(cs)
        tau_b = 1.0 if prev_sets.get("BASE") is None else leg_turnover(prev_sets["BASE"], base_set)
        prev_sets["BASE"] = base_set
        row["tau_base"] = tau_b
        for rule, (thr_main, thrs, _) in RULES.items():
            for thr in thrs:
                key = _key(rule, thr)
                mask = _rule_mask(rule, thr, feats.loc[cs]).reindex(cs).fillna(False)
                sel_set = set(pd.Index(cs)[mask])
                tau_r = (1.0 if prev_sets.get(key) is None
                         else leg_turnover(prev_sets[key], sel_set)) if sel_set else float("nan")
                if sel_set:
                    prev_sets[key] = sel_set
                row[f"n_{key}"] = len(sel_set)
                row[f"tau_{key}"] = tau_r
                for h in HORIZONS:
                    fwd = fwd_by_h[h]
                    if sel_set:
                        gross_diff = float(fwd[list(sel_set)].mean() - fwd.mean())
                        net_diff = gross_diff - (tau_r - tau_b) * row[f"cost_{h}"]
                    else:
                        gross_diff = net_diff = float("nan")
                    row[f"diff_{key}_{h}"] = gross_diff
                    row[f"net_{key}_{h}"] = net_diff
        per_period.append(row)

    res = pd.DataFrame(per_period)
    if res.empty:
        raise SystemExit("无有效实验期")
    os.makedirs(HOLDSCORE_DIR, exist_ok=True)
    res.to_json(f"{HOLDSCORE_DIR}/entry_backtest.json", orient="records",
                force_ascii=False, indent=2)             # 先落盘再报告

    print(f"=== 入场规则实验(N={len(res)}期,基础组=生产 PROD D10,主窗={MAIN_H}日登记制,"
          f"成本=两臂各自τ×round_trip)===")
    print(f"支持集损失合计 {support_loss} 只·期(四特征齐备口径,相对完整 PROD)")
    verdicts: dict[str, dict] = {}
    for rule, (thr_main, thrs, cond) in RULES.items():
        key_main = _key(rule, thr_main)
        cov = (res[f"n_{key_main}"] / res["n_base"]).mean()
        line = [f"{rule:>8}({cond})覆盖{cov:.0%} τ均{res[f'tau_{key_main}'].mean():.0%}"]
        for h in HORIZONS:
            d = res[f"net_{key_main}_{h}"].dropna()
            _, tnw, _ = newey_west_tstat(d)
            line.append(f"{h}日净 {d.mean() * 100:+.2f}%(t{tnw:+.1f})")
        print(" | ".join(line))
        d21 = res[f"net_{key_main}_{MAIN_H}"]
        d21v = d21.dropna()
        _, t21, lag21 = newey_west_tstat(d21v)
        acf1 = float(d21v.autocorr(1)) if len(d21v) > 2 else float("nan")
        # 真 LOYO(Codex P1 修复):剔除该年后在剩余样本重估符号
        years = sorted(res["year"].unique())
        loyo = []
        for y in years:
            hold_out = d21[res["year"] != y].dropna()
            v = float(hold_out.mean()) if len(hold_out) else float("nan")
            loyo.append(1 if v > 0 else (-1 if v < 0 else 0))
        reg = res["regime_up"]
        up = float(d21[reg == True].mean())    # noqa: E712
        dn = float(d21[reg == False].mean())   # noqa: E712
        nav_r = (1 + res[f"base_{MAIN_H}"] + d21.fillna(0)).cumprod()
        nav_b = (1 + res[f"base_{MAIN_H}"]).cumprod()
        mdd = lambda nav: float((nav / nav.cummax() - 1).min())   # noqa: E731
        nb = [float(res[f"net_{_key(rule, t2)}_{MAIN_H}"].dropna().mean()) for t2 in thrs]
        stats = {"prior_registered": True, "net_diff": float(d21v.mean()) * 100,
                 "loyo_signs": loyo, "up_diff": up, "dn_diff": dn, "sig_t": t21,
                 "coverage": float(cov), "mdd_rule": mdd(nav_r), "mdd_base": mdd(nav_b),
                 "neighborhood_diffs": [x * 100 for x in nb]}
        v = gate_verdict(stats)
        verdicts[rule] = {"stats": stats, "verdict": v}
        print(f"         主窗 NW lag={lag21} ACF1={acf1:+.2f} | "
              f"门禁:{'✅ 全过' if v['passed'] else '❌ ' + '/'.join(v['failed'])}")
    json.dump(verdicts, open(f"{HOLDSCORE_DIR}/entry_gate_verdicts.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    passed = [r for r, v in verdicts.items() if v["verdict"]["passed"]]
    print(f"\n结论:{'过门规则 ' + ','.join(passed) if passed else '无规则过八条门禁——生产维持诚实基线(D10+硬否决+容量,不宣称择时)'}")
    print("→ data/holdscore/entry_backtest.json + entry_gate_verdicts.json")


if __name__ == "__main__":
    main()
