"""aggressive_pick —— 股票端进攻性视图:质量地板不动 + 扣非增速排序视图 + 基金重仓主题硬排除。

背景:用户股票端要"更具进攻性",但其场外基金(华泰柏瑞质量精选/质量成长、易方达
远见成长,合计仓位大)已重仓 AI 光模块/光通信/算力硬件——股票端与基金端是同一张
资产负债表,进攻必须在**不与基金重仓主题叠加敞口**的前提下做(风险分摊)。

设计出处(每条有据,零新 magic number):
1) 质量地板 = 最新 data/holdscore/<date>_factor.json 中 decile==10 且 tier=='🟢'。
   D10 候选池是四关漏斗的既有约定(memory stock-analysis-mode),🟢=lean_tier 强干净档
   (ashare_gauntlet.record)。**进攻不等于降质量,地板不动**——只改地板之上的排序视图。
2) 进攻排序 = 池内按扣非增速 dt_netprofit_yoy 的**横截面百分位**降序(百分位复用
   factor_model.percentile_rank 既有约定,rank(pct=True) 最大=1)。诚实声明:增长强度
   **只做排序视图,不进 composite**——本地回测(memory factor-backtest-a-share)未证
   成长因子 alpha,这是用户风险偏好下的组合结构选择(股票端相对基金端更进攻),
   **非 alpha 主张**。dt_netprofit_yoy 不在 factor json 里,从本地 fina_indicator 缓存
   经 scripts.factor_rank.latest_rows(三键排序:end_date/ann_date/update_flag,取最新
   更正值)读取,不另造口径。
3) 风险分摊硬排除(定义性,来源=用户持有基金的前十大重仓所属行业):上述基金前十大
   (新易盛/中际旭创/天孚通信/长飞光纤/源杰科技/东山精密/亨通光电等)映射到 tushare
   stock_basic 行业分类 = 通信设备(新易盛/中际旭创/长飞/亨通)、元器件(天孚/东山)、
   半导体(源杰)、IT设备(算力整机外延)。⚠ EXCLUDED_INDUSTRIES 由基金持仓推导,
   **基金变了要同步改**(它是用户敞口参数,不是库常数)。另排除 data/holdings.json
   里已持有的票(已有敞口,再买=加集中度不是进攻)。
4) 右侧判定复用 ashare_gauntlet.execution.entry_readiness(缩量企稳+收复5日线,全定义性
   比较),K线经 scripts.entry_check.load_code_history 读本地 daily+adj_factor 缓存;
   单票数据不足(新上市/长停/缓存缺)打 '—' 降级不崩——排序视图不因一票瘫痪,
   与盯盘同款"单只降级、系统性问题才硬失败"分层(memory analysis-priorities)。

fail-loud 边界:factor json 缺失/D10🟢 空池/fina 缓存空/holdings.json 缺失 → 硬失败
(系统性问题);单票右侧判定数据不足 → '—'(单只降级)。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.aggressive_pick [--top 20]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from ashare_gauntlet.config import (
    CACHE_DIR as CACHE,
    HOLDINGS_PATH as HOLDINGS,
    HOLDSCORE_DIR as FACTOR_DIR,
)
from ashare_gauntlet.execution import MIN_BARS, entry_readiness
from ashare_gauntlet.factor_model import percentile_rank
from scripts.entry_check import load_code_history
from scripts.factor_rank import latest_rows

# 风险分摊硬排除集合(定义性;出处=用户场外基金前十大重仓所属 tushare 行业,见模块
# docstring 第 3 条)。⚠ 基金持仓变了要同步改这里(与 tests/test_aggressive_pick.py
# 里钉口径的测试一起改)。
EXCLUDED_INDUSTRIES: set[str] = {"通信设备", "元器件", "半导体", "IT设备"}


def quality_floor(rows: list[dict]) -> list[dict]:
    """质量地板:只留 decile==10 且 tier=='🟢'(顺序保持)。

    D10 池 + 🟢 强干净档都是既有约定(见模块 docstring 第 1 条);进攻不降质量,
    decile 缺失(None/NaN)如实不入池,不当 D10 伪造。
    """
    return [r for r in rows if r.get("decile") == 10 and r.get("tier") == "🟢"]


def aggressive_rank(d10_rows: list[dict], growth_map: dict[str, float | None],
                    excluded_industries: set[str], held_codes: set[str]) -> list[dict]:
    """进攻排序纯函数:硬排除(行业+已持仓)→ 扣非增速横截面百分位 → 降序。

    - 横截面 = **硬排除后的候选池**(被剔除的票不占分位名额,分位描述的是真候选集);
    - growth_map: ts_code → dt_netprofit_yoy(%);缺失/NaN 视为缺失,**排最后且字段
      如实 None**,不伪造 0/中位数(负增长≠缺失,负值照排);
    - 返回新行(原 dict 不就地修改),附加 dt_yoy(原值)与 growth_pct(池内百分位,
      percentile_rank 约定最大=1);空池返回 []。
    """
    held = {str(c) for c in held_codes}
    pool = [dict(r) for r in d10_rows
            if str(r.get("industry")) not in excluded_industries
            and str(r.get("ts_code")) not in held]
    if not pool:
        return []
    for r in pool:
        g = growth_map.get(str(r["ts_code"]))
        # NaN→None:float(nan)!=float(nan);缺失如实标 None,不伪造
        r["dt_yoy"] = float(g) if g is not None and float(g) == float(g) else None
    pct = percentile_rank(pd.Series([r["dt_yoy"] for r in pool], dtype=float))
    for i, r in enumerate(pool):
        p = pct.iloc[i]
        r["growth_pct"] = float(p) if p == p else None
    # 缺失排最后(sort 稳定:缺失组保持原池序);非缺失按百分位降序(=原值降序,单调)
    pool.sort(key=lambda r: (r["growth_pct"] is None,
                             -(r["growth_pct"] if r["growth_pct"] is not None else 0.0)))
    return pool


# ---------- CLI 装配层(读缓存/本地文件,fail-loud 见模块 docstring) ----------

def _latest_factor_json(out_dir: str = FACTOR_DIR) -> str:
    """最新 <YYYYMMDD>_factor.json(文件名日期即快照日;非日期命名文件不参与)。"""
    files = sorted(f for f in glob.glob(f"{out_dir}/*_factor.json")
                   if os.path.basename(f)[:8].isdigit())
    if not files:
        raise SystemExit(f"aggressive_pick: {out_dir} 无 <date>_factor.json —— 先跑 scripts.factor_rank")
    return files[-1]


def _held_codes(path: str = HOLDINGS) -> set[str]:
    """当前持仓 ts_code 集合(holdings.json 是盯盘单一真相源;缺文件硬失败,
    持仓排除是硬约束,拒绝静默跳过当作没持仓)。"""
    if not os.path.exists(path):
        raise SystemExit(f"aggressive_pick: {path} 不存在 —— 持仓排除是硬约束,拒绝静默跳过")
    with open(path, encoding="utf-8") as f:
        h = json.load(f)
    return {str(p["ts_code"]) for p in h.get("positions", [])}


def _entry_label(code: str) -> str:
    """右侧判定标签;单票K线不足/缓存缺 adj_factor → '—'(单只降级,不伪造判定)。"""
    try:
        hist = load_code_history(code, CACHE, MIN_BARS)   # 不足 MIN_BARS 会 SystemExit
    except SystemExit:
        return "—"
    try:
        return str(entry_readiness(hist["adj_close"], hist["vol"])["label"])
    except ValueError:
        return "—"


def _yoy_map(fina: pd.DataFrame, col: str) -> dict[str, float | None]:
    """fina_indicator 最新报告期某 yoy 列 → {ts_code: float|None}(NaN 如实 None)。"""
    s = pd.to_numeric(fina[col], errors="coerce")
    return {str(k): (float(v) if v == v else None) for k, v in s.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="进攻性视图:D10🟢质量地板 + 扣非增速排序视图 + 基金重仓主题硬排除")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)

    fpath = _latest_factor_json()
    with open(fpath, encoding="utf-8") as f:
        rows = json.load(f)
    pool = quality_floor(rows)
    if not pool:
        raise SystemExit(f"aggressive_pick: {fpath} 无 decile==10 且 🟢 的票 —— 上游输出异常,拒绝空排")

    # 扣非/营收增速:factor json 没有 dt_netprofit_yoy,从 fina_indicator 缓存最新报告期读
    # (latest_rows 三键排序取最新更正值,与 factor_rank 同源同口径)。
    # as_of=快照日作 PIT 闸门:快照日之后才公告的财报不得给该快照的池子排序(前视偏差,
    # 与 factor_rank 同一防线)
    as_of = os.path.basename(fpath)[:8]
    fina = latest_rows("fina_indicator", ["dt_netprofit_yoy", "tr_yoy"], as_of=as_of)
    if fina.empty:
        raise SystemExit("aggressive_pick: fina_indicator 缓存为空 —— 先跑 scripts.backfill_fina")
    growth_map = _yoy_map(fina, "dt_netprofit_yoy")
    rev_map = _yoy_map(fina, "tr_yoy")

    held = _held_codes()
    ranked = aggressive_rank(pool, growth_map, EXCLUDED_INDUSTRIES, held)
    if not ranked:
        raise SystemExit("aggressive_pick: 硬排除后候选池为空 —— 排除口径覆盖了整个 D10🟢 池,须人工复核")

    print(f"=== aggressive_pick(as_of={as_of},D10🟢地板 {len(pool)} 只 → 硬排除后 {len(ranked)} 只)===")
    print(f"硬排除行业(用户基金前十大推导,基金变了同步改):{'/'.join(sorted(EXCLUDED_INDUSTRIES))};"
          f"另排除已持仓 {len(held)} 只")
    print("排序=扣非增速横截面百分位降序(仅排序视图不进 composite:回测未证成长 alpha,"
          "系风险偏好的组合结构选择);右侧=缩量企稳+收复5日线(entry_readiness)")
    print(f"{'#':>2} {'票':<9}{'行业':<7}{'PE':>6}{'市值亿':>7}{'扣非%':>9}{'增速位':>4}"
          f"{'营收%':>7}{'MOM':>5}{'距MA20':>7} {'⚡':<2}{'右侧'}")
    pct1 = lambda x: f"{x:+.1f}" if x is not None else "—"                    # yoy 原值已是 %
    frac = lambda x: (f"{x*100:+.0f}%" if isinstance(x, (int, float)) and x == x else "—")
    nn = lambda x: (f"{x:.0f}" if isinstance(x, (int, float)) and x == x else "—")
    for i, r in enumerate(ranked[:a.top], 1):
        code = str(r["ts_code"])
        gp = r["growth_pct"]
        print(f"{i:>2} {str(r.get('name'))[:8]:<9}{str(r.get('industry'))[:6]:<7}"
              f"{nn(r.get('pe')):>6}{nn(r.get('mv')):>7}{pct1(r['dt_yoy']):>9}"
              f"{(f'{gp*100:.0f}' if gp is not None else '—'):>4}"
              f"{pct1(rev_map.get(code)):>7}{frac(r.get('MOM')):>5}{frac(r.get('dma20')):>7} "
              f"{'⚡' if r.get('spike_limit') else ' ':<2}{_entry_label(code)}")


if __name__ == "__main__":
    main()
