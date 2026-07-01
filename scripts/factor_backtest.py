"""因子 IC 回测 —— 把因子模型从"文献支撑"升级到"A股本地实证"。

对每个月度换仓日 t,用 **point-in-time**(只用 ann_date<=t 已披露财报,防未来函数)构造 6 因子横截面,
配 **T+1 开盘买入的未来收益**(forward_return_from_next_open,防用决策日自身价),算横截面 IC(秩相关)。
汇总各因子 IC均值 / ICIR(均值/标准差)/ t值 / 胜率,判断哪些因子在 A股主板真有效、该给多少权重。

因子(与 factor_rank 一致,行业中性后算 IC):EP=eps/价、BP=bps/价、ROE、GP=(营收-营业成本)/总资产、
ACC=(净利-经营现金流)/总资产(越低越好→取负)、MOM=近 mom 日前复权收益。

诚实:IC 受日线历史长度限制,是初步实证读数;样本越多越稳。

⚠️ 已知局限(对抗式审计 2026-07-01 查出,结论"低可信·不可直接采信绝对幅度",见 memory
factor-backtest-a-share):
1. 【致命·survivorship】退市股在持有期内退市→exit open=NaN→其退市前 -30~-89% 暴跌收益被
   information_coefficient 的 dropna 静默剔除;这些恰是价格崩塌后 BP 飙高的高BP价值陷阱+强负动量,
   → 系统性虚高 BP 正 IC、扭曲 MOM 符号。修法:退市按最后成交价/-100% 平仓计入 fwd,别丢 NaN。
2. 【重要·无 size 中性】缓存无 total_mv,BP 只做行业中性未做规模中性 → 无法排除"BP强=小盘/低价β代理"。
   修法:补 daily_basic/total_mv,BP 再做 size 中性。
3. 【重要·t 值虚高】t=ICIR·√N 未做 Newey-West;PIT 财务季度间月月复用 + 21日收益重叠压低 sd → t 全面高估。
   修法:自相关修正标准误/N_eff;预期 BP t 4.11→~3.0、ACC 退到勉强显著、MOM 可能跌破 |t|=2。
可信度最高的结论(即便被高估仍成立):**ROE/GP/EP = 横截面噪声**。BP 方向最扎实但幅度含虚高;
MOM 负号是 120日不跳月的 horizon artifact,换标准 12-2 动量可能翻符号——先别信。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.factor_backtest [--board main] [--fwd 21] [--mom 120]
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import pandas as pd

from ashare_gauntlet.backtest import information_coefficient
from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factor_model import industry_neutralize
from ashare_gauntlet.screen import board_of

CACHE = "data/cache"
MAIN = ("沪主板", "深主板")


def _load(ep: str, cols: list[str]) -> pd.DataFrame:
    fs = glob.glob(f"{CACHE}/{ep}/*.parquet")
    need = ["ts_code", "end_date", "ann_date"] + cols
    out = []
    for f in fs:
        try:
            out.append(pd.read_parquet(f, columns=need))
        except Exception:
            df = pd.read_parquet(f)
            out.append(df[[c for c in need if c in df.columns]])
    df = pd.concat(out, ignore_index=True)
    df["ann_date"] = df["ann_date"].astype(str)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["ts_code", "ann_date"])


def _pit(df: pd.DataFrame, asof: str) -> pd.DataFrame:
    """截至 asof 已公告(ann_date<=asof)的最新一期,每股一行(向量化 point-in-time)。"""
    d = df[df["ann_date"] <= asof]
    return d.groupby("ts_code", sort=False).tail(1).set_index("ts_code")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main")
    ap.add_argument("--fwd", type=int, default=21, help="未来收益持有天数")
    ap.add_argument("--mom", type=int, default=120, help="动量回看天数")
    a = ap.parse_args(argv)
    load_env_local()

    # 财务(全历史,带 ann_date)
    fina = _load("fina_indicator", ["eps", "bps", "roe"])
    inc = _load("income", ["revenue", "oper_cost", "n_income_attr_p"])
    cf = _load("cashflow", ["n_cashflow_act"])
    bs = _load("balancesheet", ["total_assets"])

    # 行业(行业中性用)
    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    sb = call_with_retry(lambda: pro.stock_basic(list_status="L", fields="ts_code,industry")).set_index("ts_code")
    ind_all = sb["industry"].fillna("其他")

    # 价格面板:前复权 close(算MOM/因子价) 与 open(算T+1未来收益)
    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"])
                    for f in sorted(glob.glob(f"{CACHE}/daily/*.parquet"))], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in sorted(glob.glob(f"{CACHE}/adj_factor/*.parquet"))], ignore_index=True)
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    px["aclose"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    px["aopen"] = px["open"].astype(float) * px["adj_factor"].astype(float)
    dates = sorted(px["trade_date"].unique())
    close_p = px.pivot_table(index="trade_date", columns="ts_code", values="aclose")
    open_p = px.pivot_table(index="trade_date", columns="ts_code", values="aopen")

    # 月度换仓日 = 每月最后一个交易日;需留出 mom 回看 + fwd 未来
    di = {d: i for i, d in enumerate(dates)}
    month_last: dict[str, str] = {}
    for d in dates:
        month_last[d[:6]] = d  # 同月后者覆盖 → 该月最后交易日
    rebal = [d for d in month_last.values() if di[d] >= a.mom and di[d] + 1 + a.fwd < len(dates)]

    FACTORS = ["EP", "BP", "ROE", "GP", "ACC", "MOM"]
    print(f"加载完成:{len(dates)}交易日 → {len(rebal)}个月度换仓日,逐期算 IC…", flush=True)
    ic_rows: list[dict] = []
    for k, t in enumerate(rebal):
        if k % 10 == 0:
            print(f"  …{k}/{len(rebal)}", flush=True)
        it = di[t]
        codes_price = close_p.columns[close_p.loc[t].notna()]
        codes = [c for c in codes_price if a.board != "main" or board_of(str(c)) in MAIN]
        if len(codes) < 30:
            continue
        idx = pd.Index(codes)
        price_t = close_p.loc[t, codes]
        # 未来 T+1 收益:open(t+1) → open(t+1+fwd)
        entry = open_p.iloc[it + 1][codes]
        exit_ = open_p.iloc[it + 1 + a.fwd][codes]
        fwd = (exit_ / entry - 1.0)
        # MOM
        mom = price_t / close_p.iloc[it - a.mom][codes] - 1.0
        # point-in-time 财务
        pf, pi, pc, pb = _pit(fina, t), _pit(inc, t), _pit(cf, t), _pit(bs, t)
        eps = pf["eps"].reindex(idx); bps = pf["bps"].reindex(idx); roe = pf["roe"].reindex(idx)
        rev = pi["revenue"].reindex(idx); cogs = pi["oper_cost"].reindex(idx); ni = pi["n_income_attr_p"].reindex(idx)
        ocf = pc["n_cashflow_act"].reindex(idx); ta = pb["total_assets"].reindex(idx)
        raw = pd.DataFrame(index=idx)
        raw["EP"] = eps / price_t
        raw["BP"] = bps / price_t
        raw["ROE"] = roe
        raw["GP"] = (rev - cogs) / ta
        raw["ACC"] = -((ni - ocf) / ta)   # 应计越低越好 → 取负,使高=好
        raw["MOM"] = mom
        ind = ind_all.reindex(idx).fillna("其他")
        row = {"date": t, "n": len(codes)}
        for fac in FACTORS:
            neu = industry_neutralize(raw[fac], ind)   # 行业中性(与 factor_rank 一致)
            row[fac] = information_coefficient(neu, fwd)
        ic_rows.append(row)

    res = pd.DataFrame(ic_rows)
    print(f"=== 因子 IC 回测(主板·月度换仓·未来{a.fwd}日·point-in-time·T+1收益)===")
    print(f"换仓期数 N={len(res)} | 区间 {res['date'].min()}→{res['date'].max()} | 均横截面股票数 {res['n'].mean():.0f}")
    print(f"{'因子':>5}{'IC均值':>8}{'ICIR':>7}{'t值':>7}{'胜率':>7}  解读")
    for fac in FACTORS:
        s = res[fac].dropna()
        m = s.mean(); sd = s.std(); n = len(s)
        icir = m / sd if sd else math.nan
        tstat = icir * math.sqrt(n) if n else math.nan
        hit = (s > 0).mean() * 100
        verdict = "✓强" if abs(m) > 0.03 and abs(tstat) > 2 else ("~弱有效" if abs(tstat) > 1.5 else "✗噪声")
        print(f"{fac:>5}{m:>+8.3f}{icir:>+7.2f}{tstat:>+7.2f}{hit:>6.0f}%  {verdict}")
    os.makedirs("data/holdscore", exist_ok=True)
    res.to_json("data/holdscore/factor_ic_backtest.json", orient="records", force_ascii=False, indent=2)
    print(f"→ 明细 data/holdscore/factor_ic_backtest.json")


if __name__ == "__main__":
    main()
