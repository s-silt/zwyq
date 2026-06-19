"""单股 SVG 卡(IO/CLI):读 data/cards/<as_of>.json,逐只渲染 data/cards_svg/<code>.svg。

纯渲染在 ashare_gauntlet/render_svg.py;此处只做读 json / 选码 / 落盘,贴合 cards.py 既有 pattern。
cohort 永远传全量 records(子弹图/雷达归一化的对比集)。不联网。

选码:--codes a,b,c 指定;缺省自动挑不同档代表(各档第一只,凑齐 🟢/🟡/🔴/⛔)。

Usage:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.card_svg            # 自动挑各档代表
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.card_svg --codes 601138.SH,000733.SZ
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.card_svg --as-of 20260618
"""
import glob
import json
import os
import sys
from typing import Any

from ashare_gauntlet.render_svg import render_svg_card

CARDS_DIR = "data/cards"
OUT_DIR = "data/cards_svg"
TIER_ORDER = ("🟢", "🟡", "🔴", "⛔")


def _latest_cards(as_of: str | None) -> tuple[str, list[dict[str, Any]]]:
    """读指定 as_of(或最新)的 cards json,返回 (as_of, records)。"""
    if as_of:
        path = f"{CARDS_DIR}/{as_of}.json"
    else:
        fs = sorted(glob.glob(f"{CARDS_DIR}/*.json"))
        if not fs:
            raise SystemExit(f"{CARDS_DIR} 无 cards json —— 先 scripts.cards 落库")
        path = fs[-1]
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}")
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    return os.path.basename(path)[:8], records


def _pick_default(records: list[dict[str, Any]]) -> list[str]:
    """缺省选码:每档(🟢🟡🔴⛔)挑第一只,凑不同档代表;至少 2 只。"""
    by_tier: dict[str, str] = {}
    for r in records:
        g = str(r.get("tier", {}).get("grade", ""))
        if g in TIER_ORDER and g not in by_tier:
            by_tier[g] = r["ts_code"]
    picks = [by_tier[g] for g in TIER_ORDER if g in by_tier]
    if len(picks) < 2 and records:  # 兜底:不足 2 档时补前两只
        seen = set(picks)
        for r in records:
            if r["ts_code"] not in seen:
                picks.append(r["ts_code"])
                seen.add(r["ts_code"])
            if len(picks) >= 2:
                break
    return picks


def _arg(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def main(argv: list[str]) -> None:
    as_of_arg = _arg(argv, "--as-of")
    as_of, records = _latest_cards(as_of_arg)
    by_code = {r["ts_code"]: r for r in records}

    codes_arg = _arg(argv, "--codes")
    if codes_arg:
        codes = [c.strip() for c in codes_arg.split(",") if c.strip()]
        missing = [c for c in codes if c not in by_code]
        if missing:
            raise SystemExit(f"cards({as_of}) 中无这些代码: {', '.join(missing)}")
    else:
        codes = _pick_default(records)
        grades = " ".join(f"{by_code[c]['tier']['grade']}{c}" for c in codes)
        print(f"自动挑各档代表({len(codes)}): {grades}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_paths: list[str] = []
    for code in codes:
        rec = by_code[code]
        svg = render_svg_card(rec, cohort=records)  # cohort=全量
        out = f"{OUT_DIR}/{code}.svg"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(svg)
        out_paths.append(out)
        g = rec["tier"]["grade"]
        words = (rec["tier"].get("reasons") or ["—"])[0]
        print(f"  {g} {rec['name']}({code}) entry{rec['entry']['grade']} -> {out}  ({len(svg)}B) · {words}")

    print(f"生成 {len(out_paths)} 张 SVG 于 {OUT_DIR}/ (as_of={as_of}, cohort={len(records)}只)")


if __name__ == "__main__":
    main(sys.argv[1:])
