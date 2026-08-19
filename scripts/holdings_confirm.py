"""人工确认账户状态日期:holdings.json 顶层 as_of → 指定交易日。

本脚本原则上由用户本人在终端运行(运行=人工签字)。2026-08-19 起用户追加授权:
在**用户当次主动要求更新**时,智能体可代为执行(确认即代笔,见 CLAUDE.md)——
用户发持仓截图即以截图为准,未发截图即表示该期间无未落账成交。
定时任务/后台/非用户发起的路径**仍然不得调用**。

Usage: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.holdings_confirm 20260811
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from ashare_gauntlet.account_lock import account_lock
from ashare_gauntlet.config import CACHE_DIR, HOLDINGS_PATH


def _date8(value: str) -> str:
    # 先锁 8 位数字再解析:strptime 的 %m/%d 接受不补零输入("2026811" 会被
    # 解析成 2026-08-11),放行会把 7 字符 as_of 写进账户文件,下游 _DATE8
    # 门禁将整个账户状态判非法(codex review)。
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise SystemExit(f"as_of 必须是真实 YYYYMMDD,得到 {value!r}")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise SystemExit(f"as_of 必须是真实 YYYYMMDD,得到 {value!r}")
    return value


def confirm_as_of(target: str, *, holdings_path: str = HOLDINGS_PATH,
                  cache_dir: str = CACHE_DIR) -> dict:
    """把 holdings 顶层 as_of 推进到 target;其余字段语义不变。

    fail-loud:target 非本地已知交易日(daily 分区不存在)/ as_of 倒退 /
    holdings 结构非法。target == 当前 as_of 时 no-op。返回结果摘要 dict。
    """
    _date8(target)
    if not (Path(cache_dir) / "daily" / f"{target}.parquet").exists():
        raise SystemExit(f"{target} 不是本地缓存已知的交易日(daily 分区缺失)"
                         "——先刷新 EOD 或检查日期是否敲错")
    path = Path(holdings_path)
    # 账本排他锁:与 trade_record / trade_journal --add 共用,防并发互相覆盖(codex P0)
    with account_lock(holdings_path):
        try:
            holdings = json.load(path.open(encoding="utf-8"))
        except FileNotFoundError:
            raise SystemExit(f"{holdings_path} 不存在")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{holdings_path} 不是合法 JSON: {exc}")
        if not isinstance(holdings, dict) or not isinstance(holdings.get("positions"), list):
            raise SystemExit("holdings 结构非法(需 dict 且含 positions 列表)——不修改")
        current = holdings.get("as_of")
        if current is not None:
            _date8(str(current))
            if str(current) > target:
                raise SystemExit(f"as_of 不允许倒退:当前 {current} > 目标 {target}")
            if str(current) == target:
                return {"changed": False, "as_of": target,
                        "positions": len(holdings["positions"])}

        holdings["as_of"] = target
        payload = json.dumps(holdings, ensure_ascii=False, indent=2, allow_nan=False)
        # 同目录临时文件 + 原子替换:失败不损坏原文件(holdings 是真实账户状态)
        fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix=".tmp_holdings_",
                                        dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    return {"changed": True, "as_of": target, "previous_as_of": current,
            "positions": len(holdings["positions"])}


def main(argv: "list[str] | None" = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("as_of", help="确认到的交易日 YYYYMMDD")
    a = ap.parse_args(argv)
    result = confirm_as_of(a.as_of)
    if not result["changed"]:
        print(f"as_of 已是 {result['as_of']},无需修改({result['positions']} 只持仓不变)")
        return
    print(f"as_of: {result['previous_as_of']} → {result['as_of']}"
          f"({result['positions']} 只持仓与现金字段未改动)")
    print("本次运行=人工确认期间无交易;若实际有成交,请立即手动修正持仓明细")


if __name__ == "__main__":
    main()
