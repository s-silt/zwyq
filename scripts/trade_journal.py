"""trade_journal —— 交易复盘账本(纯测量:胜率/盈亏比/期望,不打分不改排序)。

双仓制(2026-07-03 生效)配套的验证闭环:选股管线已有 pick_track 量筛选器命中率,
但"人下单"这一环零记录 —— 止损纪律有没有执行、短线仓期望是不是正的,全凭印象。
本账本把每笔进出记成结构化数据,统计只回答三个问题:
① 胜率多少;② 盈亏比(赢时平均赚 / 亏时平均亏)多少;③ 每笔期望是正是负。

口径约定(fail-loud,不伪造):
- pnl_pct 以百分点存储(+3.8 = +3.8%),win_rate 为 0~1 小数;
- pnl_pct 显式给定、不从 entry_px/exit_px 自动推导 —— 除权/手续费会让价差口径
  ≠ 账户口径(实例:宁波韵升 20260618 入 14.97 除权后 13.83,账户口径 +6.65%
  与任一价差比都对不上),推导即伪造;
- 只统计 exit_date 非空且 pnl_pct 非空的已平仓笔;有效样本 n<1 返回 {};
- 无定义的量(无亏损笔时的 payoff、全缺 hold_days 时的均值)返回 NaN 不填 0;
- 0% 平出不算赢(保守口径:名义 0 扣掉摩擦成本实为小亏);
- hold_days 为交易日口径(T+1:不含 entry 当日、含 exit 当日);它与短线仓
  "时间止损 10 交易日"(stop_policy.SHORT_TIME_STOP_DAYS_INCLUSIVE,含 entry 当日计数)
  **恒差 1 日、不是同一单位**——时间止损触发那天本账本记 9,故短线 avg_hold_days≈9
  是口径差,不能读成"提前砍仓/纪律过紧";跨口径比较前先 +1;
- approx=true 标记历史回忆口径的种子笔(价格/日期为约数),新增笔默认 false。

Usage:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m scripts.trade_journal [--bucket 短线]
    ... -m scripts.trade_journal --add code=601138.SH,bucket=短线,entry_date=20260703,entry_px=70.5,shares=200
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from ashare_gauntlet.account_lock import account_lock
from ashare_gauntlet.config import TRADE_JOURNAL_PATH as JOURNAL_PATH

# 双仓制三档:短线(≤1万,右侧入场,硬止损-7%)/ 长线(四关+财报季持有)/
# 制度前(双仓制 2026-07-03 生效之前的历史仓,只作复盘基线,不套新规)
BUCKETS = ("短线", "长线", "制度前")

# 「最近 N 笔」展示窗口:单位是**笔**不是交易日,纯展示参数,改大改小不影响
# 任何风控判定。取 10 只因为一屏够看,与 stop_policy.SHORT_TIME_STOP_DAYS_INCLUSIVE 数值
# 巧合、无派生关系——不要把这里当"制度常数副本"跟着改(反向也一样)
RECENT_N = 10

# 一笔交易的完整 schema:字段名 → 类型转换器(CLI 字符串 → 存储类型)
_FIELDS: dict[str, type] = {
    "code": str, "name": str, "bucket": str,
    "entry_date": str, "entry_px": float, "shares": int,
    "exit_date": str, "exit_px": float, "pnl_pct": float,
    "hold_days": int, "reason": str, "approx": bool,
}
# 最小必填:没有这五项就不构成一笔可复盘的交易(买了什么/哪个仓/何时/何价/多少股)
_REQUIRED = ("code", "bucket", "entry_date", "entry_px", "shares")
_DATE_RE = re.compile(r"^\d{8}$")   # tushare 通行日期格式 YYYYMMDD


def load_journal(path: str | Path = JOURNAL_PATH) -> list[dict]:
    """读账本。顶层必须是 {"trades": [...]} —— 形状不对 fail-loud,不猜不修。"""
    data = json.load(open(path, encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("trades"), list):
        raise ValueError(f"{path} 顶层应为 {{'trades': [...]}},实际 {type(data).__name__} —— "
                         f"账本形状坏了,拒绝静默兼容")
    return data["trades"]


def save_journal(trades: list[dict], path: str | Path = JOURNAL_PATH) -> None:
    """写账本(整体覆盖,保持 {"trades": [...]} 形状,中文原样落盘)。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"trades": trades}, f, ensure_ascii=False, indent=2)
        f.write("\n")


def stats(trades: list[dict], bucket: str | None = None) -> dict:
    """胜率/盈亏比/期望统计。只算 exit_date 与 pnl_pct 都非空的已平仓笔。

    返回 {n, win_rate, avg_win_pct, avg_loss_pct, payoff, avg_hold_days, expectancy};
    有效样本 n<1 返回 {}(不伪造空样本统计)。
    - payoff(盈亏比)= avg_win / |avg_loss|;无赢/无亏/|avg_loss|=0 → NaN;
    - expectancy = win_rate*avg_win + (1-win_rate)*avg_loss ≡ 全部 pnl 的均值
      (空类概率权重为 0,恒等式对无赢/无亏样本同样成立,按均值算免 NaN 传染);
    - avg_hold_days 只对 hold_days 非空的笔取均值,全缺 → NaN。
    - **expectancy_w / win_rate_w = 按 shares 加权**(codex P1-3):部分减仓
      (trade_record --trim)会让"一行=一个完整仓位"不再成立——先减 100 股 +30%、
      再清 1100 股 −10%,按笔等权得 win_rate 50%/expectancy +10%,而实际加权 ≈ −6.7%。
      既有按笔字段语义保持不变(历史可比),加权字段是新增读数;shares 缺失的笔不
      计入加权(w_n 另报),全缺 → NaN,不用 1 顶替(不伪造权重)。
    """
    if bucket is not None:
        trades = [t for t in trades if t.get("bucket") == bucket]
    closed = [t for t in trades
              if t.get("exit_date") is not None and t.get("pnl_pct") is not None]
    n = len(closed)
    if n < 1:
        return {}
    pnl = [float(t["pnl_pct"]) for t in closed]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]   # 0% 计入亏:不赢即输(见模块 docstring)
    avg_win = sum(wins) / len(wins) if wins else math.nan
    avg_loss = sum(losses) / len(losses) if losses else math.nan
    payoff = avg_win / abs(avg_loss) if wins and losses and avg_loss != 0 else math.nan
    hold = [t["hold_days"] for t in closed if t.get("hold_days") is not None]
    # shares 加权(部分减仓后"一行=一个完整仓位"不再成立;shares 缺失的笔不参与)
    weighted = [(float(t["pnl_pct"]), float(t["shares"])) for t in closed
                if isinstance(t.get("shares"), int) and not isinstance(t.get("shares"), bool)
                and t["shares"] > 0]
    w_sum = sum(w for _, w in weighted)
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "payoff": payoff,
        "avg_hold_days": sum(hold) / len(hold) if hold else math.nan,
        "expectancy": sum(pnl) / n,
        "n_w": len(weighted),
        "expectancy_w": (sum(p * w for p, w in weighted) / w_sum) if w_sum else math.nan,
        "win_rate_w": (sum(w for p, w in weighted if p > 0) / w_sum) if w_sum else math.nan,
    }


def parse_add(spec: str) -> dict:
    """解析 --add 的 "k=v,k=v" 串为一笔完整交易(缺省字段补 None,schema 恒齐全)。

    fail-loud:未知字段 / 缺必填 / bucket 不在三档 / 日期非 YYYYMMDD / approx 非
    true|false,一律 ValueError —— 打错字不能静默生成脏记录。
    注意值内不能含半角逗号(reason 用中文标点)。
    """
    t: dict = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"--add 片段 {item!r} 不是 k=v 形式")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in _FIELDS:
            raise ValueError(f"--add 未知字段 {k!r},合法字段:{sorted(_FIELDS)}")
        if k == "approx":
            low = v.strip().lower()
            if low not in ("true", "false"):
                raise ValueError(f"approx 只接受 true/false,收到 {v!r}")
            t[k] = low == "true"
        else:
            t[k] = _FIELDS[k](v)
    missing = [k for k in _REQUIRED if k not in t]
    if missing:
        raise ValueError(f"--add 缺必填字段 {missing}(必填:{list(_REQUIRED)})")
    if t["bucket"] not in BUCKETS:
        raise ValueError(f"bucket 必须是 {BUCKETS} 之一,收到 {t['bucket']!r}")
    for dk in ("entry_date", "exit_date"):
        if t.get(dk) is not None and not _DATE_RE.match(t[dk]):
            raise ValueError(f"{dk} 应为 YYYYMMDD,收到 {t[dk]!r}")
    # 缺省补齐:schema 恒 12 字段;approx 默认 False(历史种子才 approx)
    for k in _FIELDS:
        t.setdefault(k, False if k == "approx" else None)
    return t


def _fmt(x: float | None, suffix: str = "") -> str:
    """NaN/None 显示为 —(无定义就摆出来,不装成 0)。"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.2f}{suffix}" if suffix else f"{x:.2f}"


def _print_stats_table(trades: list[dict], only_bucket: str | None) -> None:
    rows = [(b, stats(trades, bucket=b)) for b in BUCKETS
            if only_bucket is None or b == only_bucket]
    if only_bucket is None:
        rows.append(("总体", stats(trades)))
    print(f"{'仓':<4}{'笔数':>4}{'胜率':>8}{'均赢':>9}{'均亏':>9}"
          f"{'盈亏比':>7}{'均持有(交易日)':>10}{'期望/笔':>9}")
    for name, s in rows:
        if not s:
            print(f"{name:<4}{'0':>4}{'—':>8}(无已平仓且有盈亏的样本)")
            continue
        print(f"{name:<4}{s['n']:>4}{s['win_rate']*100:>7.0f}%"
              f"{_fmt(s['avg_win_pct'], '%'):>9}{_fmt(s['avg_loss_pct'], '%'):>9}"
              f"{_fmt(s['payoff']):>7}{_fmt(s['avg_hold_days']):>10}"
              f"{_fmt(s['expectancy'], '%'):>9}")


def _print_recent(trades: list[dict], only_bucket: str | None) -> None:
    pool = [t for t in trades if only_bucket is None or t.get("bucket") == only_bucket]
    # 最近 = 按最后动作日排序(有 exit 用 exit_date,持有中用 entry_date)
    pool = sorted(pool, key=lambda t: t.get("exit_date") or t["entry_date"])[-RECENT_N:]
    print(f"\n最近 {min(len(pool), RECENT_N)} 笔(~ 为回忆口径 approx):")
    for t in pool:
        mark = "~" if t.get("approx") else " "
        exit_s = (f"{t['exit_date']}@{t['exit_px']}" if t.get("exit_date") and t.get("exit_px") is not None
                  else (f"{t['exit_date']}@?" if t.get("exit_date") else "持有中"))
        pnl_s = _fmt(float(t["pnl_pct"]), "%") if t.get("pnl_pct") is not None else "—"
        hold_s = f"{t['hold_days']}d" if t.get("hold_days") is not None else "—"
        print(f" {mark}[{t['bucket']}] {t.get('name') or '':　<5}{t['code']:<10}"
              f"{t['entry_date']}@{t['entry_px']} ×{t['shares']} → {exit_s:<16}"
              f"{pnl_s:>9} {hold_s:>4}  {t.get('reason') or ''}")


def main(argv: list[str] | None = None, path: str | Path = JOURNAL_PATH) -> None:
    ap = argparse.ArgumentParser(prog="trade_journal", description="交易复盘账本:分仓胜率/盈亏比/期望 + 最近笔录")
    ap.add_argument("--bucket", choices=BUCKETS, help="只看指定仓")
    ap.add_argument("--add", metavar="k=v,k=v,...",
                    help=f"追加一笔并落盘;必填 {','.join(_REQUIRED)};值内勿用半角逗号")
    args = ap.parse_args(argv)

    if args.add:
        # 账本排他锁(与 trade_record / holdings_confirm 共用,锁文件=同目录
        # holdings.lock):锁覆盖"读→改→写"全程,防并发写互相覆盖(codex P0)
        with account_lock(Path(path).parent / "holdings.json"):
            trades = load_journal(path)
            t = parse_add(args.add)
            trades.append(t)
            save_journal(trades, path)
        print(f"已追加:[{t['bucket']}] {t['code']} {t['entry_date']}@{t['entry_px']} "
              f"×{t['shares']}(共 {len(trades)} 笔)\n")
    else:
        trades = load_journal(path)

    _print_stats_table(trades, args.bucket)
    _print_recent(trades, args.bucket)
    print("\n(口径:pnl 为账户口径百分点、显式记录不由价差推导;持有=交易日 T+1;"
          "0% 计亏;甬金类 pnl 未记录笔与持有中笔不进统计)")


if __name__ == "__main__":
    main()
