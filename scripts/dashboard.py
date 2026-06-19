"""C 层(IO/CLI):读 data/cards/<as_of>.json,调纯渲染 render_dashboard,
写 data/panels/dashboard_<as_of>.html(自含单文件多股总览决策台)。

纯渲染逻辑在 ashare_gauntlet/render_html.py(无 IO);此处只做读 json / 选最新 as_of /
写产物文件,贴合 scripts/panel.py、scripts/card_svg.py 既有 pattern。cohort 即全量 records。
不联网。

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


def main(as_of: str | None = None) -> None:
    resolved = as_of or _latest_as_of()
    records = _load_cards(resolved)

    html = render_dashboard(records)

    os.makedirs(PANELS_DIR, exist_ok=True)
    out_path = f"{PANELS_DIR}/dashboard_{resolved}.html"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"总览决策台 -> {out_path}（{len(records)} 只，{len(html)} 字符）")


if __name__ == "__main__":
    argv = sys.argv[1:]
    ao: str | None = None
    if "--as-of" in argv:
        ao = argv[argv.index("--as-of") + 1]
    main(ao)
