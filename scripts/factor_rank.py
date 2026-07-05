"""横截面因子排序(B 路)—— 用 factor_model 对全主板做"质量×价值"因子打分,替掉数值持有分。

为什么是这套(对照审计 scoring-needs-theory):每个因子映射公认 anomaly、全程零 magic number。
- 入分因子(COMPOSITE_FACTORS):EP=1/PE(Basu 1977/FF 价值)、BP=1/PB(FF 1992)、
  ACC=(归母净利−经营现金流)/总资产(Sloan 1996 应计,**越低越好**=低应计高质量)——
  三者均经本地 12.5 年回测(N=149,成本后+剔壳稳健)验证。
- 展示因子(不入分):ROE、GP/A=(营收−营业成本)/总资产(Novy-Marx 2013)——12 年噪声
  (t<1)+ 互相冗余(0.63);MOM——反转向且成本后为负。保留 f_ 列供判断层参考。
- 方法:每因子 行业+市值双中性 → 横截面百分位 → 等权合成 → 十分位(见 factor_model)。
- 过滤:lean_tier 剔 🔴(三降/亏损),避免给恶化业务做"便宜"排序=价值陷阱;只对 🟢🟡 排。

- 标注(同为零 magic number):⚡脉冲=近5个交易日内触及过涨停(daily.high 对照 stk_limit
  涨停价——交易所规则常数,见 touched_limit_up),直接服务"新强名先问是不是涨停顶上来的"铁律;
  趋势=MOM 当日横截面 top decile(复用 to_decile 十分位既有约定)。⚡ 优先于趋势展示,
  JSON 输出 spike_limit 布尔列;此前的 ret5>15% / MOM>20% 手拍阈值已废除(审计点名)。

诚实边界:① 三因子等权仍是无信息先验(拒绝用同一份回测拟合权重=数据窥探);ACC 成本后
月差为负(其价值在信息正交,非独立可交易);② tushare 行业较细、
小行业内中位数去均值偏噪;③ 金融业各因子语义特殊,已行业中性(同业内比)但跨行业合成仍需
模式优先判断兜(见 model-aware-judgment);④ 输出十分位/分位,不报 1 分粒度绝对分(去伪精度)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.factor_rank [--board main] [--industry kw] [--top 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from ashare_gauntlet.data.env import load_env_local
from ashare_gauntlet.data.fetch import call_with_retry, fetch_market_day
from ashare_gauntlet.data.partition import assert_adj_complete, date_partition_files
from ashare_gauntlet.data.tushare_source import make_pro_api
from ashare_gauntlet.factor_model import (
    composite,
    factor_percentile,
    momentum_return,
    to_decile,
    touched_limit_up,
)
from ashare_gauntlet.record import lean_tier
from ashare_gauntlet.screen import board_of
from scripts.backfill_fina import expected_min_end_date

CACHE = "data/cache"
OUT_DIR = "data/holdscore"
MAIN = ("沪主板", "深主板")

# 入分因子 = 本地 12.5 年实证过的三个(N=149, 2014-2026, 成本后+剔壳稳健):
# EP t4.9/BP t6.8 可交易有效,ACC t3.0 且与一切正交。ROE/GP 12 年噪声(t<1)
# + 互相冗余(相关 0.63)+ ROE 半被 EP 吸收(0.49)→ 降为展示列(与 MOM 同待遇),
# 不再以"文献先验"名义占据 2/5 权重稀释被验证的信号(见 memory factor-backtest-a-share)。
COMPOSITE_FACTORS = ("f_EP", "f_BP", "f_ACC")


def composite_score(factor_cols: pd.DataFrame) -> pd.Series:
    """三因子等权合成(EP+BP+ACC)。展示列(f_ROE/f_GPOA/f_MOM)不入分。"""
    return composite(factor_cols[list(COMPOSITE_FACTORS)])


def composite_inputs_complete(df: pd.DataFrame) -> pd.Series:
    """入池门槛:三个入分因子**原值**(EP/BP/ACC)全齐。

    展示列(roe/GPOA/MOM)不参与门槛——旧"至少4/5因子"门槛在 composite 改三因子后
    产生两类错误(外部 review 点名):EP/BP/ACC 齐但 ROE+GPOA 缺 → 错杀;ACC 缺但
    ROE/GPOA 齐 → 错放(score 实际只剩 EP/BP,与他票不是同一数学对象)。要求全齐
    (而非 ≥2/3)是定义性选择:每只入榜票的 score 必须是同一个三因子等权对象,
    composite 的 skipna 均值只兜"极端缺列仍出数"的下游安全,不作为入池许可。
    """
    return df[["EP", "BP", "ACC"]].notna().all(axis=1)


def latest_rows(endpoint: str, cols: list[str], as_of: str) -> pd.DataFrame:
    """每只取该表 **as_of 时点已公告** 的最新报告期一行(ts_code 为索引)。

    as_of(YYYYMMDD)是 PIT 闸门:先过滤 ann_date<=as_of 再取最新——价格缓存停在
    旧日而财务缓存已刷到新公告时,不带闸门会拿"未来财报"排旧日横截面(前视偏差)。
    ann_date 缺失(NaN)的行无法证明其时点合法性,同样被过滤(字符序 'nan'/'None'
    恒大于 8 位日期,PIT 从严);某票 as_of 时点无任何已公告行 → 如实不入横截面。
    """
    # ann_date/update_flag 参与排序:同 (end_date) 多行(快照 vs 更正重述)时必须确定性地取
    # 最新更正值——update_flag='1' 是 tushare 更正后记录(官方语义),字符序恰排最后;
    # 否则不稳定排序会随机取到更正前的错值(实测 600115 归母净利更正前后差 1.6 亿,直接进 ACC 分子)。
    need = ["ts_code", "end_date", "ann_date", "update_flag"] + cols
    out: dict[str, pd.Series] = {}
    for f in glob.glob(f"{CACHE}/{endpoint}/*.parquet"):
        try:
            df = pd.read_parquet(f, columns=need)
        except Exception:
            df = pd.read_parquet(f)
            df = df[[c for c in need if c in df.columns]]
        if df.empty:
            continue
        if "ann_date" in df.columns:
            df = df[df["ann_date"].astype(str) <= str(as_of)]   # PIT:只用 as_of 已公告的行
            if df.empty:
                continue
        sort_keys = [k for k in ("end_date", "ann_date", "update_flag") if k in df.columns]
        out[str(df.iloc[0]["ts_code"])] = df.sort_values(sort_keys, kind="mergesort").iloc[-1]
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

    pro = make_pro_api(os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"])
    # 全历史日线+复权 → 前复权价(算动量 MOM / 近5日收益 / 距MA20)。
    # 走 date_partition_files:只认 ^\d{8}\.parquet$(daily/ 实际混入过整段拉取文件,
    # 直接 glob 会污染交易日历与面板,见 data.partition 模块 docstring)
    da = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
                    for f in date_partition_files(CACHE, "daily")], ignore_index=True)
    aj = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "adj_factor"])
                    for f in date_partition_files(CACHE, "adj_factor")], ignore_index=True)
    as_of = str(da["trade_date"].max())
    px = da.merge(aj, on=["ts_code", "trade_date"], how="left")
    # fail-loud ①:adj_factor 缺日会让全市场该日 adj_close=NaN 被静默 dropna 吞掉(MOM/MA20 悄悄错位)
    assert_adj_complete(px)

    # 财务横截面必须晚于 as_of 计算并以其为 PIT 闸门:价格缓存停在旧日而财务已刷到
    # 新公告时,ann_date>as_of 的"未来财报"不得进入 as_of 日的横截面(前视偏差)
    fina = latest_rows("fina_indicator", ["roe", "netprofit_yoy", "dt_netprofit_yoy", "tr_yoy", "ocfps"],
                       as_of=as_of)
    inc = latest_rows("income", ["revenue", "oper_cost", "n_income_attr_p"], as_of=as_of)
    cf = latest_rows("cashflow", ["n_cashflow_act"], as_of=as_of)
    bs = latest_rows("balancesheet", ["total_assets"], as_of=as_of)

    # fail-loud ②:财务缓存新鲜度——法定披露期后必须已含新报告期(否则财报季后整条管线静默用旧财报)
    _expected = expected_min_end_date(as_of)
    _fina_max = str(fina["end_date"].astype(str).max()) if "end_date" in fina.columns and len(fina) else ""
    if _fina_max < _expected:
        raise SystemExit(f"财务缓存过期:横截面最新报告期 {_fina_max or '(空)'} < 法定应披露 {_expected}——"
                         f"先跑 python -m scripts.backfill_fina --mode core --refresh")
    px["adj_close"] = px["close"].astype(float) * px["adj_factor"].astype(float)
    mom_d: dict[str, float | None] = {}
    ret5_d: dict[str, float | None] = {}
    last_d: dict[str, float] = {}
    dma20_d: dict[str, float | None] = {}
    for code, g in px.sort_values("trade_date").groupby("ts_code"):
        ac = g["adj_close"].dropna()
        cl = g["close"].astype(float)
        mom_d[str(code)] = momentum_return(ac, 120)
        ret5_d[str(code)] = float(ac.iloc[-1] / ac.iloc[-6] - 1.0) if len(ac) >= 6 else None
        last_d[str(code)] = float(cl.iloc[-1])
        # 距MA20 用**前复权价**算(XD 除息日的分红缺口不是跌——未复权会假破位,实盘已两次踩中)
        dma20_d[str(code)] = float(ac.iloc[-1] / ac.tail(20).mean() - 1.0) if len(ac) >= 20 else None
    # ⚡脉冲定义性锚:近5个交易日 high 对照 stk_limit 涨停价(交易所规则常数,fetch_market_day
    # 自动按日缓存),触及即标——涨停触及是规则事件,替代 ret5>15% 手拍阈值
    last5 = sorted(str(d) for d in px["trade_date"].unique())[-5:]
    hi5 = pd.concat([fetch_market_day(pro, "daily", d, CACHE)[["ts_code", "trade_date", "high"]]
                     for d in last5], ignore_index=True)
    lim5 = pd.concat([fetch_market_day(pro, "stk_limit", d, CACHE)[["ts_code", "trade_date", "up_limit"]]
                      for d in last5], ignore_index=True)
    spike_codes = touched_limit_up(hi5, lim5)
    db = fetch_market_day(pro, "daily_basic", as_of, CACHE).set_index("ts_code")  # 缓存版:全字段落盘,一份缓存服务所有下游
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
    df["MOM"] = pd.Series(mom_d).reindex(idx)        # 动量:~6月前复权收益(仅展示,不入分)
    df["ret5"] = pd.Series(ret5_d).reindex(idx)      # 近5日收益(仅展示;⚡改用触及涨停定义性锚)
    df["last"] = pd.Series(last_d).reindex(idx)
    df["dma20"] = pd.Series(dma20_d).reindex(idx)    # 距MA20(前复权口径,防XD假破位)
    df["spike_limit"] = [str(i) in spike_codes for i in idx]  # ⚡近5交易日内触及过涨停

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
    # 入池门槛:三个入分因子原值全齐(展示列不参与门槛,见 composite_inputs_complete)
    df = df[composite_inputs_complete(df)]

    ind = df["industry"]
    logmv = np.log(df["mv"].where(df["mv"] > 0))   # size 中性:与回测 _neutralize 同一形态
    df["f_EP"] = factor_percentile(df["EP"], ind, higher_is_better=True, logmv=logmv)
    df["f_BP"] = factor_percentile(df["BP"], ind, higher_is_better=True, logmv=logmv)
    df["f_ROE"] = factor_percentile(df["roe"], ind, higher_is_better=True, logmv=logmv)
    df["f_GPOA"] = factor_percentile(df["GPOA"], ind, higher_is_better=True, logmv=logmv)
    df["f_ACC"] = factor_percentile(df["ACC"], ind, higher_is_better=False, logmv=logmv)  # 应计越低越好
    df["f_MOM"] = factor_percentile(df["MOM"], ind, higher_is_better=True, logmv=logmv)   # 仅展示列
    # 三因子等权合成 EP+BP+ACC(行业+市值双中性,与 factor_backtest 被验证的形态一致)。
    # ROE/GP 不入分:12 年噪声(t<1)+ 互相冗余 0.63 + ROE 半被 EP 吸收;MOM 不入分:反转向
    # 且成本后为负——三者均保留 f_ 列作展示/判断层参考(见 COMPOSITE_FACTORS 注)。
    df["score"] = composite_score(df)
    df["decile"] = to_decile(df["score"])
    # 趋势标注 = MOM 当日横截面 top decile(复用 to_decile 十分位既有约定,零新常数;
    # 替代 MOM>20% 手拍阈值)。NaN(历史不足)不标。
    df["mom_top"] = to_decile(df["MOM"]).eq(10).fillna(False).astype(bool)
    df = df.sort_values("score", ascending=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    keep = ["name", "industry", "tier", "pe", "pb", "roe", "mv", "decile", "score",
            "f_EP", "f_BP", "f_ROE", "f_GPOA", "f_ACC", "f_MOM", "MOM", "ret5", "last", "dma20",
            "spike_limit"]
    df[keep].reset_index().rename(columns={"index": "ts_code"}).to_json(
        f"{OUT_DIR}/{as_of}_factor.json", orient="records", force_ascii=False, indent=2)

    print(f"=== 横截面因子排序(EP+BP+ACC 三因子入分·行业+市值双中性·ROE/GP/MOM 仅展示, as_of={as_of}, {len(df)}只)→ {OUT_DIR}/{as_of}_factor.json ===")
    print("分位=双中性百分位(0-100);入分三因子=12.5年实证(N=149,成本后+剔壳稳健);距MA20=前复权口径(防XD假破位)")
    print("⚡脉冲=近5交易日内触及过涨停(stk_limit 规则价,优先展示);趋势=MOM 横截面 top decile(十分位约定)")
    print(f"{'#':>2} {'D':>2} {'档':>2} {'票':<9}{'行业':<7}{'EP':>3}{'BP':>3}{'ROE':>4}{'GP':>3}{'ACC':>4}{'MOM':>4}{'PE':>5}{'市值亿':>7} {'位置/趋势'}")
    pc = lambda x: f"{x*100:.0f}" if isinstance(x, (int, float)) and x == x else "—"
    nn = lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and x == x else "—"
    for i, (code, r) in enumerate(df.head(a.top).iterrows(), 1):
        dma20, mom = r.get("dma20"), r.get("MOM")
        dma = f"{dma20*100:+.0f}%" if isinstance(dma20, (int, float)) and dma20 == dma20 else "—"
        # ⚡(触及涨停,规则事件)优先于趋势(MOM top decile);mom_top=True 蕴含 MOM 非 NaN
        spike = "⚡脉冲(触涨停)" if bool(r.get("spike_limit")) else (
            f"趋势{mom*100:+.0f}%" if bool(r.get("mom_top")) else "")
        print(f"{i:>2} {(int(r['decile']) if pd.notna(r['decile']) else 0):>2} {r['tier']:>2} "
              f"{str(r['name'])[:8]:<9}{str(r['industry'])[:6]:<7}"
              f"{pc(r['f_EP']):>3}{pc(r['f_BP']):>3}{pc(r['f_ROE']):>4}{pc(r['f_GPOA']):>3}{pc(r['f_ACC']):>4}{pc(r['f_MOM']):>4}"
              f"{nn(r['pe']):>5}{nn(r.get('mv')):>7} 距MA20{dma} {spike}")


if __name__ == "__main__":
    main()
