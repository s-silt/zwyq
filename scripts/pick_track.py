"""pick_track —— 筛选器命中率闭环(纯测量,不打分、不改排序)。

审计点名的最大验证缺口:回测只有 N=40 单一 regime,factor_rank 的样本外有效性只能靠
"记录每期 D10 → 追踪后续真实收益" 积累证据;此前快照一直在攒(data/holdscore/*_factor.json)
但零脚本回读——筛选器实盘命中率处于零证据状态。本脚本补上这个闭环:

① diff:最近两期 factor json 的 D10 进/出名单(**新进票**正是动量陷阱高危,呼应
   memory momentum-screen-limitup"新强名先问是不是涨停顶上来的");
② 前向收益:对每期历史快照的 D10,算 快照日→今天 的前复权收益,对比全主板等权基准
   (超额 = 命中率证据;纯测量,几个月后回答"这筛选器到底选得准不准")。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.pick_track
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

import pandas as pd

CACHE = "data/cache"
OUT_DIR = "data/holdscore"


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


def main() -> None:
    snaps = sorted(glob.glob(f"{OUT_DIR}/*_factor.json"))
    if not snaps:
        raise SystemExit("无 factor 快照(先跑 scripts.factor_rank)")

    # 价格面板(前复权)
    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
                    for f in sorted(glob.glob(f"{CACHE}/daily/*.parquet"))], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in sorted(glob.glob(f"{CACHE}/adj_factor/*.parquet"))], ignore_index=True)
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    px["adj_close"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    latest = str(px["trade_date"].max())

    def load_d10(path: str) -> list[str]:
        rows = json.load(open(path, encoding="utf-8"))
        return [r["ts_code"] for r in rows if r.get("decile") == 10]

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

    # ② 各历史快照 D10 的前向收益 vs 全主板等权基准(命中率证据,纯测量)
    print(f"\n=== D10 前向收益 vs 全主板等权(至 {latest};样本外命中率证据,逐期积累)===")
    print(f"{'快照日':>10}{'D10只数':>7}{'D10均收益':>10}{'基准均收益':>10}{'超额':>8}")
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
        print(f"{snap:>10}{len(d10):>7}{d10_m*100:>+9.1f}%{uni_m*100:>+9.1f}%{(d10_m-uni_m)*100:>+7.1f}%")
    print("(超额>0=筛选器跑赢自己的宇宙;样本少时噪声大,别过度解读单期)")


if __name__ == "__main__":
    main()
