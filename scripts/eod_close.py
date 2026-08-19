"""收盘一条命令:刷新行情 → 人工签字确认日期 → 决策管线 → 刷新估值 → 每日一屏。

**顺序不是随意排的**(实测踩过死锁):`holdings_confirm` 要求目标日是本地缓存已知
交易日 → 刷新必须在签字前;而 `buy_list` 要求账户 as_of == 行情日 → 决策必须在签字后。
把整条 eod_ops 放在签字之前,buy_list 必然卡在账户日期门禁,管线 break 后连签字都
走不到。第 ③ 步的 eod_ops 会再跑一次 refresh(幂等、缓存命中即返回),不重复取数。

把散在四条命令里的收盘动作串成一条,但**人工签字环节不被自动化掉**:
`holdings_confirm` 的语义是"运行本脚本 = 人工确认该期间无未落账成交",这个确认
不能因为串进编排就悄悄替用户做了。因此本脚本:
- 必须**显式传日期**(不自动取今天:自动取会让"确认哪一天"这件事失去人的判断);
- 推进 as_of 前在 **TTY 交互**里要求输入该日期作签字短语,非 TTY(定时任务/管道/
  智能体)一律拒绝推进——**定时任务不得代人签字**;
- 若账户 as_of 已等于目标日期(无需推进),则不需要签字,可在任意环境跑完其余步骤。

失败即停(fail-loud):任一步非零退出都不继续,避免在残缺数据上产出"看起来正常"的
一屏。退出码沿用 daily_brief 语义:0=平静,2=有需人工处理事项,1=数据/管线失败。

Usage(收盘后、当日 EOD 数据已发布时跑):
    E:\\zwyq\\.venv\\Scripts\\python.exe -m scripts.eod_close 20260819
    E:\\zwyq\\.venv\\Scripts\\python.exe -m scripts.eod_close 20260819 --skip-eod
    (或 .\\scripts\\q.ps1 close 20260819)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ashare_gauntlet.config import HOLDINGS_PATH

STEP_TIMEOUT = 1800


def _date8(value: str) -> str:
    if not re.fullmatch(r"\d{8}", value or ""):
        raise SystemExit(f"日期必须是 8 位 YYYYMMDD,得到 {value!r}")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise SystemExit(f"{value} 不是真实日期")
    return value


def _run(module: str, args: "list[str] | None" = None) -> int:
    """跑一步,输出直通终端(用户要看得到各步真实反馈)。返回退出码。"""
    cmd = [sys.executable, "-m", module, *(args or [])]
    print(f"\n$ python -m {module} {' '.join(args or [])}", flush=True)
    try:
        return subprocess.run(cmd, timeout=STEP_TIMEOUT, check=False).returncode
    except subprocess.TimeoutExpired:
        print(f"!! {module} 超时(>{STEP_TIMEOUT}s)", file=sys.stderr)
        return 124


def current_as_of(holdings_path: str = HOLDINGS_PATH) -> "str | None":
    """只读账户当前 as_of;读不到返回 None(不猜)。"""
    try:
        data = json.loads(Path(holdings_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("as_of") if isinstance(data, dict) else None
    return str(value) if value else None


def confirm_prompt(target: str, previous: "str | None", *,
                   stream=None, is_tty: "bool | None" = None) -> bool:
    """人工签字:要求在 TTY 里逐字输入目标日期。非 TTY 一律拒绝(不得代签)。

    分离出来是为了可测:测试注入 stream/is_tty,不依赖真实终端。
    """
    # isatty 在本平台不可靠(实测 Git Bash on Windows 下 `< /dev/null` 也报 True),
    # 故**不把它当唯一防线**:两道都要过——① 必须是交互终端;② 读到的内容必须逐字
    # 等于目标日期。管道喂日期会被 ① 挡下(实测 isatty=False),重定向空输入会被 ②
    # 挡下。任一不满足都不推进账本,失败方向恒为"不签"。
    interactive = sys.stdin.isatty() if is_tty is None else is_tty
    manual_hint = (f"请在终端手动跑:  python -m scripts.holdings_confirm {target}")
    if not interactive:
        print("!! 需要推进账户 as_of,但当前不是交互终端——定时任务/管道/智能体"
              "不得代人签字。" + manual_hint, file=sys.stderr)
        return False
    print(f"\n=== 人工确认 ===\n账户 as_of: {previous or '(缺失)'} → {target}")
    print("此确认表示:该期间**没有未落账的成交**(若有,先跑 trade_record 落账再来)。")
    print(f"确认请逐字输入日期 {target}(其他任意输入=取消):")
    reader = stream if stream is not None else sys.stdin
    raw = reader.readline()
    answer = (raw or "").strip()
    if not answer:
        # 读到 EOF/空行:多半是非交互环境被 isatty 误判为终端
        print(f"未收到签字输入(EOF/空行)——已取消,账户未改动。{manual_hint}",
              file=sys.stderr)
        return False
    if answer != target:
        print(f"输入 {answer!r} ≠ {target},已取消(账户未改动)")
        return False
    return True


def main(argv: "list[str] | None" = None) -> None:
    ap = argparse.ArgumentParser(description="收盘编排:EOD → 人工确认 → 估值 → 一屏")
    ap.add_argument("as_of", help="收盘交易日 YYYYMMDD(显式给出,不自动取今天)")
    ap.add_argument("--skip-eod", action="store_true",
                    help="跳过 eod_ops(定时任务 17:30 已跑过时用)")
    a = ap.parse_args(argv)
    target = _date8(a.as_of)

    # ① 先刷行情:holdings_confirm 要求目标日是**本地缓存已知交易日**,所以刷新必须
    #    在签字之前;但 buy_list 又要求账户 as_of == 行情日(require_account_as_of),
    #    所以决策不能和刷新绑在一起跑——顺序必须是 刷新 → 签字 → 决策。
    #    (实测踩坑:把整条 eod_ops 放在签字前,buy_list 必然卡在账户日期门禁,
    #     管线 break 后连签字都走不到,形成死锁。)
    if not a.skip_eod:
        code = _run("scripts.refresh")
        if code != 0:
            raise SystemExit(f"refresh 失败(退出码 {code})——行情未更新,不在残缺数据上继续")

    # ② 人工签字 + 推进账户日期(as_of 已对齐则跳过,无需签字)
    previous = current_as_of()
    if previous == target:
        print(f"\n账户 as_of 已是 {target},无需确认")
    else:
        if not confirm_prompt(target, previous):
            raise SystemExit(2)
        code = _run("scripts.holdings_confirm", [target])
        if code != 0:
            raise SystemExit(f"holdings_confirm 失败(退出码 {code})")

    # ③ 账户日期对齐后再跑决策管线(退出码 2=有状态变化,属正常)
    if not a.skip_eod:
        code = _run("scripts.eod_ops")
        if code not in (0, 2):
            raise SystemExit(f"eod_ops 失败(退出码 {code})——先修数据/管线,不在残缺数据上继续")

    # ④ EOD 估值快照(时间止损/持仓风险数字的前置)
    code = _run("scripts.holdings_watch", [target])
    if code != 0:
        raise SystemExit(f"holdings_watch 失败(退出码 {code})")

    # ④ 每日一屏:退出码直通(0 平静 / 2 有待办 / 1 系统不可信)
    raise SystemExit(_run("scripts.daily_brief"))


if __name__ == "__main__":
    main()
