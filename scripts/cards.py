"""D 编排层(IO/CLI):读缓存全市场日线 + 实时 stock_basic/daily_basic,逐只 build_record,
落库 data/cards/<as_of>.json;有 data/factcheck/<as_of>.json 则回写;有上一交易日 cards 则打印跨日 diff。

纯函数在 ashare_gauntlet/record.py;此处只做 IO/编排/打印,贴合 screen.py/fundamentals.py 既有 pattern。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.cards [--limit N] [--no-render]
落库 data/cards/<as_of>.json,默认顺带渲染诚实面板(.md)+观察名单总览(.html)到 data/panels/
(render_outputs 复用 A/C 渲染器);--no-render 只出数据。单股 SVG 卡另由 scripts.card_svg 生成。
"""
import glob
import json
import os
import sys
from typing import Any, cast

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR as CACHE, CARDS_DIR, PANELS_DIR, WATCHLIST_PATH as WATCHLIST, tushare_pro
from ashare_gauntlet.data.fetch import call_with_retry, fetch_symbol_table
from ashare_gauntlet.factsheet import market_returns
from ashare_gauntlet.record import build_record, diff_records, merge_factcheck
from ashare_gauntlet.render_html import render_dashboard
from ashare_gauntlet.render_md import render_md

SYMBOL_TABLES = (
    "income", "fina_indicator", "balancesheet", "cashflow", "share_float",
    "pledge_stat", "stk_holdertrade", "namechange", "forecast", "express",
)


def _load(ep: str) -> pd.DataFrame:
    fs = glob.glob(f"{CACHE}/{ep}/*.parquet")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _prev_cards(as_of: str) -> dict[str, Any] | None:
    """上一交易日 cards,按 ts_code 索引,用于 diff。"""
    fs = sorted(glob.glob(f"{CARDS_DIR}/*.json"))
    prev = [f for f in fs if os.path.basename(f)[:8] < as_of]
    if not prev:
        return None
    with open(prev[-1], encoding="utf-8") as fh:
        data = json.load(fh)
    return {r["ts_code"]: r for r in data}


def dump_cards(records: list[dict[str, Any]], out_path: str) -> None:
    """落库 cards JSON,``allow_nan=False`` 兜底防线(#1)。

    正常情况下 record.py 的技术面叶子已过 ``_num``(NaN→None),不该有 NaN 漏到这里;
    但 ``json.dump`` 默认会把 NaN 写成非法字面量 ``NaN``(非标准 JSON,下游/外部解析器
    会失败或静默读成错值)。``allow_nan=False`` 让任何漏网的 NaN/Infinity **响亮抛
    ValueError** 而非写出脏数据 —— 数据源纯净优先于"先落个库"。
    """
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2, allow_nan=False)


def render_outputs(records: list[dict[str, Any]], as_of: str, panels_dir: str = PANELS_DIR) -> dict[str, str]:
    """从内存 records 直出 markdown 诚实面板。

    **HTML 决策台已退役**(2026-08-19 用户拍板):它原本的核心用途是把"🟢×entry 档"
    换算成可照抄的买入金额清单,而 entry 择时档已被 methodology §11 实证否决、且该路径
    不过 D10/composite/fact-check/治理/行业上限任一守卫;金额已先行删除,剩余的"观察
    名单总览"与 daily_brief(q today)+ MCP 完全重叠。多一个入口只会让状态不一致、
    并诱使后人恢复旧金额逻辑。历史产物只读留档(见 render_html 的废弃水印)。
    """
    os.makedirs(panels_dir, exist_ok=True)
    md_path = f"{panels_dir}/{as_of}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_md(records))
    return {"md": md_path}


def main(limit: int | None = None, render: bool = True) -> None:
    daily, adj = _load("daily"), _load("adj_factor")
    if daily.empty:
        raise SystemExit("data/cache/daily 为空 —— 先 backfill")
    as_of = str(daily["trade_date"].max())
    mr = market_returns(daily, adj, (5, 20))

    with open(WATCHLIST, encoding="utf-8") as fh:
        watch = json.load(fh)
    if limit:
        watch = watch[:limit]

    # 语义统一(2026-07 Codex 审查确认的有意行为修正):本文件旧私有 _load_env_local
    # 用 setdefault(进程环境优先),与 env.py 权威 override 语义相反,属历史不一致;
    # 收敛后统一为 .env.local 权威覆盖(与其余全部脚本一致)。
    pro = tushare_pro()
    records: list[dict[str, Any]] = []
    for i, item in enumerate(watch):
        code = item["ts_code"]
        fund_tables = {t: fetch_symbol_table(pro, t, code, CACHE) for t in SYMBOL_TABLES}
        sbr = call_with_retry(lambda c=code: pro.stock_basic(ts_code=c, fields="ts_code,name,industry"))
        dbr = call_with_retry(lambda c=code: pro.daily_basic(ts_code=c, trade_date=as_of, fields="ts_code,pe_ttm,pb,total_mv"))
        name = item.get("name", "") or ""
        industry = "-"
        if sbr is not None and not sbr.empty:
            name = str(sbr.iloc[0]["name"])
            ind = sbr.iloc[0]["industry"]
            industry = str(ind) if pd.notna(ind) else "-"
        db_row = dbr.iloc[0] if dbr is not None and not dbr.empty else None
        daily_sub = cast(pd.DataFrame, daily[daily["ts_code"] == code])
        adj_sub = cast(pd.DataFrame, adj[adj["ts_code"] == code])
        rec = build_record(
            code, name=name, industry=industry, as_of=as_of,
            daily_sub=daily_sub, adj_sub=adj_sub,
            mr=mr, fund_tables=fund_tables, db_row=db_row,
        )
        rec["theme"] = item.get("theme", "")
        rec["tags"] = item.get("tags", [])
        records.append(rec)
        pe = rec["valuation"]["pe_ttm"]
        pe_s = f"PE{pe:.0f}" if pe is not None else "PE-"
        print(f"[{i + 1}/{len(watch)}] {rec['tier']['grade']} {name}({code}) entry{rec['entry']['grade']} {pe_s}")

    # factcheck 回写(铁律:merge_factcheck 不覆盖接口数字)
    fc_path = f"data/factcheck/{as_of}.json"
    if os.path.exists(fc_path):
        with open(fc_path, encoding="utf-8") as fh:
            fcs = json.load(fh)
        records = [merge_factcheck(r, fcs[r["ts_code"]]) if r["ts_code"] in fcs else r for r in records]
        print(f"factcheck 回写 {sum(1 for r in records if r.get('factcheck'))} 只")

    os.makedirs(CARDS_DIR, exist_ok=True)
    out_path = f"{CARDS_DIR}/{as_of}.json"
    dump_cards(records, out_path)
    print(f"落库 {out_path}({len(records)} 只)")

    if render:
        out = render_outputs(records, as_of)
        print(f"渲染 {out['md']}(HTML 决策台已退役,日常入口=q today)")

    prev = _prev_cards(as_of)
    if prev:
        print("\n== 跨日变化 ==")
        for r in records:
            old = prev.get(r["ts_code"])
            if old is None:
                continue
            d = diff_records(old, r)
            if d["tier_change"] or d["new_flags"] or d["dropped_flags"]:
                print(f"  {r['name']}({r['ts_code']}): tier{d['tier_change']} +{d['new_flags']} -{d['dropped_flags']}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    lim: int | None = None
    if "--limit" in argv:
        lim = int(argv[argv.index("--limit") + 1])
    main(lim, render="--no-render" not in argv)
