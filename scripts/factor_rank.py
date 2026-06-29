"""横截面因子排序(B 路)—— 用 factor_model 对全主板做"质量×价值"因子打分,替掉数值持有分。

为什么是这套(对照审计 scoring-needs-theory):每个因子映射公认 anomaly、全程零 magic number。
- 因子:EP=1/PE(Basu 1977/FF 价值)、BP=1/PB(FF 1992)、ROE(盈利能力)、
  GP/A=(营收−营业成本)/总资产(Novy-Marx 2013 gross profitability)、
  ACC=(归母净利−经营现金流)/总资产(Sloan 1996 应计,**越低越好**=低应计高质量)。
- 方法:每因子 行业内中位数去均值 → 横截面百分位 → 等权合成 → 十分位(见 factor_model)。
- 过滤:lean_tier 剔 🔴(三降/亏损),避免给恶化业务做"便宜"排序=价值陷阱;只对 🟢🟡 排。

诚实边界:① 仅 2026Q1 一个财报横截面,无法回测 IC → 等权(不伪造权重);② tushare 行业较细、
小行业内中位数去均值偏噪;③ 金融业各因子语义特殊,已行业中性(同业内比)但跨行业合成仍需
模式优先判断兜(见 model-aware-judgment);④ 输出十分位/分位,不报 1 分粒度绝对分(去伪精度)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.factor_rank [--board main] [--industry kw] [--top 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factor_model import composite, factor_percentile, momentum_return, to_decile
from ashare_gauntlet.record import lean_tier
from ashare_gauntlet.screen import board_of

CACHE = "data/cache"
OUT_DIR = "data/holdscore"
MAIN = ("沪主板", "深主板")


def latest_rows(endpoint: str, cols: list[str]) -> pd.DataFrame:
    """每只取该表最新报告期一行(ts_code 为索引)。"""
    need = ["ts_code", "end_date"] + cols
    out: dict[str, pd.Series] = {}
    for f in glob.glob(f"{CACHE}/{endpoint}/*.parquet"):
        try:
            df = pd.read_parquet(f, columns=need)
        except Exception:
            df = pd.read_parquet(f)
            df = df[[c for c in need if c in df.columns]]
        if df.empty:
            continue
        out[str(df.iloc[0]["ts_code"])] = df.sort_values("end_date").iloc[-1]
    return pd.DataFrame(out).T


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main", choices=("main", "all"))
    ap.add_argument("--industry", default=None)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)
    load_env_local()

    fina = latest_rows("fina_indicator", ["roe", "netprofit_yoy", "dt_netprofit_yoy", "tr_yoy", "ocfps"])
    inc = latest_rows("income", ["revenue", "oper_cost", "n_income_attr_p"])
    cf = latest_rows("cashflow", ["n_cashflow_act"])
    bs = latest_rows("balancesheet", ["total_assets"])

    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    # 全历史日线+复权 → 前复权价(算动量 MOM / 近5日脉冲 / 距MA20)
    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
                    for f in sorted(glob.glob(f"{CACHE}/daily/*.parquet"))], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in sorted(glob.glob(f"{CACHE}/adj_factor/*.parquet"))], ignore_index=True)
    as_of = str(da["trade_date"].max())
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    px["adj_close"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    mom_d: dict[str, float | None] = {}
    ret5_d: dict[str, float | None] = {}
    last_d: dict[str, float] = {}
    ma20_d: dict[str, float | None] = {}
    for code, g in px.sort_values("trade_date").groupby("ts_code"):
        ac = g["adj_close"].dropna()
        cl = g["close"].astype(float)
        mom_d[str(code)] = momentum_return(ac, 120)
        ret5_d[str(code)] = float(ac.iloc[-1] / ac.iloc[-6] - 1.0) if len(ac) >= 6 else None
        last_d[str(code)] = float(cl.iloc[-1])
        ma20_d[str(code)] = float(cl.tail(20).mean()) if len(cl) >= 20 else None
    db = call_with_retry(lambda: pro.daily_basic(trade_date=as_of, fields="ts_code,pe_ttm,pb,total_mv")).set_index("ts_code")
    sb = call_with_retry(lambda: pro.stock_basic(list_status="L", fields="ts_code,name,industry")).set_index("ts_code")

    idx = fina.index
    df = pd.DataFrame(index=idx)
    df["name"] = sb["name"].reindex(idx)
    df["industry"] = sb["industry"].reindex(idx).fillna("其他")
    df["pe"] = _num(db["pe_ttm"].reindex(idx))
    df["pb"] = _num(db["pb"].reindex(idx))
    df["mv"] = _num(db["total_mv"].reindex(idx)) / 1e4
    df["roe"] = _num(fina["roe"])
    df["np_yoy"] = _num(fina["netprofit_yoy"])
    df["dedt_yoy"] = _num(fina["dt_netprofit_yoy"])
    df["rev_yoy"] = _num(fina["tr_yoy"])
    df["ocfps"] = _num(fina["ocfps"])
    df["MOM"] = pd.Series(mom_d).reindex(idx)        # 动量:~6月前复权收益
    df["ret5"] = pd.Series(ret5_d).reindex(idx)      # 近5日收益(脉冲检测)
    df["last"] = pd.Series(last_d).reindex(idx)
    df["ma20px"] = pd.Series(ma20_d).reindex(idx)

    rev = _num(inc["revenue"].reindex(idx))
    cogs = _num(inc["oper_cost"].reindex(idx))
    ni = _num(inc["n_income_attr_p"].reindex(idx))
    ocf = _num(cf["n_cashflow_act"].reindex(idx))
    ta = _num(bs["total_assets"].reindex(idx))

    # 5 因子原值(EP/BP 仅在盈利/正净资产下有定义)
    df["EP"] = (1.0 / df["pe"]).where(df["pe"] > 0)
    df["BP"] = (1.0 / df["pb"]).where(df["pb"] > 0)
    df["GPOA"] = (rev - cogs) / ta
    df["ACC"] = (ni - ocf) / ta

    df["tier"] = [lean_tier(df["np_yoy"].iat[k], df["dedt_yoy"].iat[k], df["rev_yoy"].iat[k],
                            ocfps=df["ocfps"].iat[k], roe=df["roe"].iat[k]) for k in range(len(df))]

    if a.board == "main":
        df = df[[board_of(str(i)) in MAIN for i in df.index]]
    if a.industry:
        df = df[df["industry"].astype(str).str.contains(a.industry, na=False)]
    df = df[df["tier"].isin(["🟢", "🟡"])]            # 剔 🔴(恶化),只对干净/瑕疵档做因子排序
    df = df[~df["name"].astype(str).str.contains("ST", na=False)]  # 剔退市风险警示(定义性)
    df = df[df["pe"] > 0]                             # 仅盈利:EP/BP 价值因子在 E>0 下才有定义
    # 至少 4/5 因子可得才排(避免少数因子撑起的不可靠合成)
    fcols0 = [df["EP"], df["BP"], df["roe"], df["GPOA"], df["ACC"]]
    df = df[sum(c.notna() for c in fcols0) >= 4]

    ind = df["industry"]
    df["f_EP"] = factor_percentile(df["EP"], ind, higher_is_better=True)
    df["f_BP"] = factor_percentile(df["BP"], ind, higher_is_better=True)
    df["f_ROE"] = factor_percentile(df["roe"], ind, higher_is_better=True)
    df["f_GPOA"] = factor_percentile(df["GPOA"], ind, higher_is_better=True)
    df["f_ACC"] = factor_percentile(df["ACC"], ind, higher_is_better=False)  # 应计越低越好
    df["f_MOM"] = factor_percentile(df["MOM"], ind, higher_is_better=True)   # 动量:越强越好(Carhart MOM)
    # 6 因子等权合成:价值(EP/BP)+ 质量(ROE/GP/ACC)+ 动量(MOM)。动量补回后,
    # 洁美这类"强趋势+真成长但高PE"不再被价值×质量埋到底。
    df["score"] = composite(df[["f_EP", "f_BP", "f_ROE", "f_GPOA", "f_ACC", "f_MOM"]])
    df["decile"] = to_decile(df["score"])
    df = df.sort_values("score", ascending=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    keep = ["name", "industry", "tier", "pe", "pb", "roe", "decile", "score",
            "f_EP", "f_BP", "f_ROE", "f_GPOA", "f_ACC", "f_MOM", "MOM", "ret5", "last", "ma20px"]
    df[keep].reset_index().rename(columns={"index": "ts_code"}).to_json(
        f"{OUT_DIR}/{as_of}_factor.json", orient="records", force_ascii=False, indent=2)

    print(f"=== 横截面因子排序(价值×质量×动量·行业中性·零magic number, as_of={as_of}, {len(df)}只)→ {OUT_DIR}/{as_of}_factor.json ===")
    print("分位=行业内百分位(0-100,越高越好);ACC=低应计;MOM=6月动量;趋势/脉冲:近5日涨>15%标⚡脉冲(防低基数炒作)")
    print(f"{'#':>2} {'D':>2} {'档':>2} {'票':<9}{'行业':<7}{'EP':>3}{'BP':>3}{'ROE':>4}{'GP':>3}{'ACC':>4}{'MOM':>4}{'PE':>5} {'位置/趋势'}")
    pc = lambda x: f"{x*100:.0f}" if isinstance(x, (int, float)) and x == x else "—"
    nn = lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and x == x else "—"
    for i, (code, r) in enumerate(df.head(a.top).iterrows(), 1):
        last, ma20px, ret5, mom = r.get("last"), r.get("ma20px"), r.get("ret5"), r.get("MOM")
        dma = f"{(last/ma20px-1)*100:+.0f}%" if isinstance(last, (int, float)) and isinstance(ma20px, (int, float)) and ma20px else "—"
        spike = "⚡脉冲" if isinstance(ret5, (int, float)) and ret5 > 0.15 else (
            f"趋势{mom*100:+.0f}%" if isinstance(mom, (int, float)) and mom > 0.2 else "")
        print(f"{i:>2} {(int(r['decile']) if pd.notna(r['decile']) else 0):>2} {r['tier']:>2} "
              f"{str(r['name'])[:8]:<9}{str(r['industry'])[:6]:<7}"
              f"{pc(r['f_EP']):>3}{pc(r['f_BP']):>3}{pc(r['f_ROE']):>4}{pc(r['f_GPOA']):>3}{pc(r['f_ACC']):>4}{pc(r['f_MOM']):>4}"
              f"{nn(r['pe']):>5} 距MA20{dma} {spike}")


if __name__ == "__main__":
    main()
