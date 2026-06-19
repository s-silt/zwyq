"""C 层(IO/CLI):读 data/cards/<as_of>.json,调纯渲染 render_dashboard,
写 data/panels/dashboard_<as_of>.html(自含单文件多股总览决策台)。

纯渲染逻辑在 ashare_gauntlet/render_html.py(无 IO);此处只做读 json / 选最新 as_of /
写产物文件,贴合 scripts/panel.py、scripts/card_svg.py 既有 pattern。cohort 即全量 records。
不联网。容错放本 IO 层:坏 record 渲染失败则跳过并把 ts_code 响亮报到 stderr(纯层
fail-loud 不吞错;不纯数据 surface 不藏),好票照常成表。

Usage:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.dashboard            # 取 data/cards 下最新
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.dashboard --as-of 20260618
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

from ashare_gauntlet.render_html import render_dashboard
from ashare_gauntlet.render_svg import render_svg_card

CARDS_DIR = "data/cards"
PANELS_DIR = "data/panels"


def _latest_as_of() -> str:
    """data/cards 下最新一份 cards 的 as_of(文件名前 8 位)。"""
    fs = sorted(glob.glob(f"{CARDS_DIR}/*.json"))
    if not fs:
        raise SystemExit(f"{CARDS_DIR} 下没有 cards json —— 先 scripts.cards 落库")
    return os.path.basename(fs[-1])[:8]


def _load_cards(as_of: str) -> list[dict[str, Any]]:
    path = f"{CARDS_DIR}/{as_of}.json"
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise SystemExit(f"{path} 不是 record 数组")
    return data


def _partition_renderable(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """容错层:逐只试渲染单股 SVG 卡,把渲染失败(编程错误/坏 record)的票剔出、连同
    ts_code+错误响亮返回,好票照常进 dashboard。

    铁律(见 memory analysis-priorities):纯渲染层(render_html/render_svg)保持 fail-loud、
    不吞错;容错只放这一层,且失败被 surface(调用方打印到 stderr)、绝不静默吞——一只坏
    record 不炸整张表,但不纯数据被显式暴露、可排查。返回 (good, failures)。
    """
    good: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for r in records:
        try:
            render_svg_card(r, cohort=records)
        except Exception as exc:  # IO 层有意容错:剔除坏票但响亮上报(非静默吞)
            failures.append((str(r.get("ts_code", "?")), f"{type(exc).__name__}: {exc}"))
            continue
        good.append(r)
    return good, failures


def main(as_of: str | None = None) -> None:
    resolved = as_of or _latest_as_of()
    records = _load_cards(resolved)

    good, failures = _partition_renderable(records)
    if failures:
        print(
            f"!! {len(failures)}/{len(records)} 只 record 渲染失败,已跳过(数据需排查):",
            file=sys.stderr,
        )
        for code, err in failures:
            print(f"   - {code}: {err}", file=sys.stderr)

    html = render_dashboard(good)

    os.makedirs(PANELS_DIR, exist_ok=True)
    out_path = f"{PANELS_DIR}/dashboard_{resolved}.html"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"总览决策台 -> {out_path}（{len(good)}/{len(records)} 只渲染,{len(html)} 字符）")


if __name__ == "__main__":
    argv = sys.argv[1:]
    ao: str | None = None
    if "--as-of" in argv:
        ao = argv[argv.index("--as-of") + 1]
    main(ao)
