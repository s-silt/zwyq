"""人工成交的结构化落账:一行命令把真实成交写进 holdings + trade_journal。

本脚本是**人工维护动作的工具化**(与 holdings_confirm 同边界):必须由用户
本人在终端运行,运行本身即"该笔成交真实发生"的人工签字——智能体与 MCP
服务器均不得调用。所有数字(股数/净额/成交价/账户口径盈亏)必须由用户从
券商成交回报抄进参数;脚本只做机械账务、校验与幂等,不推导不估算
(pnl_pct 显式抄券商值——除权/手续费使价差口径≠账户口径,推导即伪造,
见 trade_journal 口径约定)。

卖出(仅支持整仓清盘;同日拆单请把成交回报汇总成一笔;部分卖出手工编辑):
    python -m scripts.trade_record --sell 600875.SH --date 20260817 \
        --net 31250 --exit-px 26.5 --pnl-pct 4.2 --reason C2
买入(新建仓;name/industry/last 从当日 factor snapshot 机器数据补齐):
    python -m scripts.trade_record --buy 600694.SH --date 20260818 \
        --shares 700 --cost-px 9.87 --net 6915 --bucket long

reason 枚举(P-04 整改代码化:规则动作不得写成人工判断):
    C2=系统规则(C2)  STOP=系统规则(止损)  MANUAL=人工判断

事务边界(codex 复审后的明确取舍):两个 JSON 各自原子替换(tempfile+
os.replace),写前对**两份完整 payload 预序列化并校验**(allow_nan=False),
故序列化/校验失败零副作用;残余风险仅剩两次 os.replace 之间的进程级中断,
发生时脚本打印精确恢复指引。跨文件单事务(SQLite 账本)= 改变全仓存储契约,
登记为后续独立任务,不在本工具内偷做。运行期持排他锁文件,禁止并发落账。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from ashare_gauntlet.account_lock import account_lock
from ashare_gauntlet.config import (
    CACHE_DIR,
    HOLDINGS_PATH,
    HOLDSCORE_DIR,
    TRADE_JOURNAL_PATH,
)
from ashare_gauntlet.data.partition import date_partition_files
from scripts.trade_journal import _FIELDS as JOURNAL_FIELDS

REASONS = {"C2": "系统规则(C2)", "STOP": "系统规则(止损)", "MANUAL": "人工判断"}
BUCKET_TO_JOURNAL = {"short": "短线", "long": "长线"}   # journal 仓别=中文三档契约
NET_TOLERANCE = 0.05


class TradeRecordError(SystemExit):
    """落账前置校验失败——账户文件一个字节都未改动。"""


def _reject_constant(name: str):
    raise TradeRecordError(f"账户文件含非标准 JSON 常量 {name}(NaN/Infinity)——先修复再落账")


def _load_strict(path: str, what: str) -> dict:
    try:
        data = json.load(open(path, encoding="utf-8"), parse_constant=_reject_constant)
    except FileNotFoundError:
        raise TradeRecordError(f"{path} 不存在——{what} 缺失,不落账")
    except json.JSONDecodeError as exc:
        raise TradeRecordError(f"{path} 不是合法 JSON: {exc}——先修复再落账")
    if not isinstance(data, dict):
        raise TradeRecordError(f"{path} 顶层必须是对象——不落账")
    return data


def _date8(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise TradeRecordError(f"{field} 必须是真实 YYYYMMDD,得到 {value!r}")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise TradeRecordError(f"{field} 必须是真实 YYYYMMDD,得到 {value!r}")
    return value


def _positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TradeRecordError(f"{field} 必须是正有限数,得到 {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise TradeRecordError(f"{field} 必须是正有限数,得到 {value!r}")
    return number


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TradeRecordError(f"{field} 必须是有限数,得到 {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise TradeRecordError(f"{field} 必须是有限数,得到 {value!r}")
    return number


def _whole_shares(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TradeRecordError(f"{field} 必须是正整数股数,得到 {value!r}(小数股拒绝截断)")
    if value <= 0:
        raise TradeRecordError(f"{field} 必须是正整数股数,得到 {value!r}")
    return value


def _require_trading_day(date: str, cache_dir: str, field: str) -> str:
    """复用 holdings_confirm 的门禁:date 必须是本地缓存已知交易日。"""
    if not (Path(cache_dir) / "daily" / f"{date}.parquet").exists():
        raise TradeRecordError(f"{field}={date} 不是本地缓存已知交易日(daily 分区缺失)"
                               "——周末/节假日/未来日期或未刷新 EOD")
    return date


def _trading_hold_days(entry_date: str, exit_date: str, cache_dir: str) -> int:
    """交易日口径(T+1:不含 entry 当日、含 exit 当日)——journal 契约。

    比 holdings.held_trading_days(含 entry 当日,风控/时间止损侧)恒小 1:时间止损
    触发的那一笔在本口径记 9 而非 10。两侧各用各的,不互相换算(见 trade_journal
    模块口径约定),对齐数字前先确认在看哪一套。

    日历源=本地 daily 分区。分区缺口会静默少算(codex 二轮),故加护栏:
    区间内相邻分区自然日间隔 >15 天(A股连续休市上限≈国庆/春节 ~8 天)
    视为缓存缺段,fail-loud 要求先回填,不产出可能错误的 hold_days。
    """
    if entry_date > exit_date:
        raise TradeRecordError(f"entry_date {entry_date} 晚于 exit_date {exit_date}——日期错")
    if entry_date == exit_date:
        # A股 T+1:当日买入不可当日卖出,hold_days=0 必然是日期抄错(codex P2-6)
        raise TradeRecordError(f"entry_date 与 exit_date 同为 {exit_date}——A股 T+1 当日"
                               "买入不可卖出,核对买入日/成交日")
    days = sorted(os.path.basename(f)[:8] for f in date_partition_files(cache_dir, "daily"))
    if entry_date not in days:
        raise TradeRecordError(f"entry_date {entry_date} 不在本地交易日历——核对买入日")
    span = [d for d in days if entry_date <= d <= exit_date]
    for prev, cur in zip(span, span[1:]):
        gap = (datetime.strptime(cur, "%Y%m%d") - datetime.strptime(prev, "%Y%m%d")).days
        if gap > 15:
            raise TradeRecordError(f"本地交易日历在 {prev}→{cur} 存在 {gap} 天缺口"
                                   "——daily 缓存缺段,先回填再落账(hold_days 拒绝少算)")
    return len(span) - 1


def _journal_bucket(position_bucket: object) -> str:
    mapped = BUCKET_TO_JOURNAL.get(str(position_bucket), str(position_bucket))
    if mapped not in ("短线", "长线", "制度前"):
        raise TradeRecordError(f"持仓 bucket={position_bucket!r} 无法映射到 journal 三档"
                               "(短线/长线/制度前)——先修正持仓字段")
    return mapped


def _validate_journal_shape(trade: dict) -> dict:
    """按 trade_journal._FIELDS 契约核形状与类型;字段集必须恰好一致。"""
    if set(trade) != set(JOURNAL_FIELDS):
        raise TradeRecordError(f"journal 行字段集不符契约: {sorted(set(trade) ^ set(JOURNAL_FIELDS))}")
    for key, caster in JOURNAL_FIELDS.items():
        value = trade[key]
        if value is not None and not isinstance(value, caster):
            raise TradeRecordError(f"journal 字段 {key} 类型应为 {caster.__name__},得到 {value!r}")
    return trade


def _atomic_write(path: str, text: str) -> None:
    target = Path(path)
    fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix=f".tmp_{target.stem}_",
                                    dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _validate_holdings(holdings: dict, path: str) -> None:
    if not isinstance(holdings.get("positions"), list):
        raise TradeRecordError(f"{path} 缺 positions 列表——不落账")
    _finite(holdings.get("cash"), "holdings.cash")
    if float(holdings["cash"]) < 0:
        raise TradeRecordError(f"holdings.cash={holdings['cash']!r} 为负——账本已坏,先人工修复")
    for p in holdings["positions"]:
        if not isinstance(p, dict) or not p.get("ts_code"):
            raise TradeRecordError("positions 含非法行——不落账")
        _whole_shares(p.get("shares"), f"持仓 {p.get('ts_code')} shares")


def record_sell(ts_code: str, *, date: str, net: float, exit_px: float,
                pnl_pct: float, reason_key: str, entry_date: "str | None" = None,
                approx: bool = False, advance_as_of: bool = False,
                holdings_path: str = HOLDINGS_PATH,
                journal_path: str = TRADE_JOURNAL_PATH,
                cache_dir: str = CACHE_DIR) -> dict:
    """整仓卖出落账。所有校验先行,任何失败=零副作用;写入=先 journal 后 holdings。"""
    net = _positive(net, "--net")
    exit_px = _positive(exit_px, "--exit-px")
    pnl_pct = _finite(pnl_pct, "--pnl-pct")
    if reason_key not in REASONS:
        raise TradeRecordError(f"--reason 必须是 {sorted(REASONS)} 之一")
    with account_lock(holdings_path):
        holdings = _load_strict(holdings_path, "账户状态")
        journal = _load_strict(journal_path, "交易流水")
        _validate_holdings(holdings, holdings_path)
        if not isinstance(journal.get("trades"), list):
            raise TradeRecordError("journal.trades 结构非法——不落账")
        exit_date = _require_trading_day(_date8(date, "--date"), cache_dir, "--date")
        as_of = _date8(holdings.get("as_of"), "holdings.as_of")
        if exit_date < as_of:
            raise TradeRecordError(f"--date {exit_date} 早于账户 as_of {as_of}"
                                   "——账户 as_of 不可倒退,核对成交日")
        if exit_date != as_of and not advance_as_of:
            raise TradeRecordError(f"--date {exit_date} ≠ 账户 as_of {as_of}——先跑 "
                                   "holdings_confirm 推进账户日期再落账,或加 --advance-as-of "
                                   "在本次落账同一事务内推进(=确认该区间内无其他成交)")

        matches = [p for p in holdings["positions"] if p.get("ts_code") == ts_code]
        if len(matches) != 1:
            raise TradeRecordError(f"{ts_code} 在持仓中匹配到 {len(matches)} 条(需恰好 1 条)——不落账")
        position = matches[0]
        shares = _whole_shares(position.get("shares"), "持仓 shares")
        # 同日"先 --trim 减半、后 --sell 清仓剩余"是合法序列,不能只按 (code,date) 判重;
        # 完全相同的 (股数,价) 才是重复落账(股数守恒由"清仓=持仓全量"天然保证)
        for t in journal["trades"]:
            if (t.get("code") == ts_code and str(t.get("exit_date")) == exit_date
                    and t.get("shares") == shares
                    and _finite(t.get("exit_px"), "journal exit_px") == exit_px):
                raise TradeRecordError(
                    f"journal 已有 {ts_code} exit_date={exit_date} {shares}股@{exit_px} "
                    "的同一笔流水——重复落账被拒绝")
        # 清仓后若仍留着该股的 active SELL 条件单,重新买入时旧单会再次生效造成误卖
        # (codex P1:record_sell 此前完全不查条件单)
        _co_s = holdings.get("conditional_orders")
        _orders_s = (_co_s.get("orders") if isinstance(_co_s, dict)
                     else (_co_s if isinstance(_co_s, list) else None))
        if isinstance(_orders_s, list):
            stale = [str(o.get("order_id") or o.get("ts_code")) for o in _orders_s
                     if isinstance(o, dict) and o.get("ts_code") == ts_code
                     and str(o.get("side", "")).upper() == "SELL"
                     and str(o.get("status", "")).lower() == "active"]
            if stale:
                raise TradeRecordError(
                    f"{ts_code} 清仓但仍有 active SELL 条件单 {stale}——先在券商撤单并更新"
                    "holdings.conditional_orders 再落账(留着会在重新建仓后误触发)")
        gross = shares * exit_px
        # 方向性护栏(codex P1):卖出净回款只会被费用削减,不可能高于 gross
        if not (gross * (1 - NET_TOLERANCE) <= net <= gross):
            raise TradeRecordError(f"--net {net:,.2f} 应在 [{gross * (1 - NET_TOLERANCE):,.2f}, "
                                   f"{gross:,.2f}](股数×价−费)内——疑似抄错,核对成交回报")
        resolved_entry = entry_date or position.get("entry_date")
        if not resolved_entry:
            raise TradeRecordError("持仓无 entry_date 且未提供 --entry-date——journal 契约要求必填;"
                                   "历史仓请补 --entry-date(约数需加 --approx)")
        resolved_entry = _date8(resolved_entry, "entry_date")

        trade = _validate_journal_shape({
            "code": ts_code, "name": str(position.get("name") or ts_code),
            "bucket": _journal_bucket(position.get("bucket")),
            "entry_date": resolved_entry,
            "entry_px": _positive(position.get("cost"), f"持仓 {ts_code} cost"),
            "shares": shares, "exit_date": exit_date, "exit_px": exit_px,
            "pnl_pct": pnl_pct,
            "hold_days": _trading_hold_days(resolved_entry, exit_date, cache_dir),
            "reason": REASONS[reason_key], "approx": bool(approx),
        })
        new_journal = {**journal, "trades": [*journal["trades"], trade]}
        new_holdings = {**holdings,
                        "as_of": exit_date,  # 前进式推进(--advance-as-of 时)或维持;与落账同一原子写
                        "positions": [p for p in holdings["positions"]
                                      if p.get("ts_code") != ts_code],
                        "cash": round(float(holdings["cash"]) + net, 2),
                        "closed": [*holdings.get("closed", []),
                                   {"ts_code": ts_code, "name": trade["name"],
                                    "note": f"{exit_date} 清仓,理由={trade['reason']}(trade_record)"}]}
        # 预序列化两份 payload(allow_nan=False):序列化失败零副作用(codex P0)
        journal_text = json.dumps(new_journal, ensure_ascii=False, indent=2, allow_nan=False)
        holdings_text = json.dumps(new_holdings, ensure_ascii=False, indent=2, allow_nan=False)
        _atomic_write(journal_path, journal_text)
        try:
            _atomic_write(holdings_path, holdings_text)
        except BaseException:
            print(f"!!! journal 已写入但 holdings 替换失败——中间态!恢复指引:"
                  f"删除 {journal_path} 中 code={ts_code} exit_date={exit_date} 的最后一行,"
                  f"或手工完成 holdings 修改(移除 {ts_code},cash+={net})后勿重跑")
            raise
    return {"side": "SELL", "ts_code": ts_code, "shares": shares, "net": net,
            "pnl_pct": pnl_pct, "cash": new_holdings["cash"],
            "positions": len(new_holdings["positions"])}


def record_trim(ts_code: str, *, date: str, shares: int, net: float, exit_px: float,
                pnl_pct: float, reason_key: str, entry_date: "str | None" = None,
                approx: bool = False, advance_as_of: bool = False,
                holdings_path: str = HOLDINGS_PATH,
                journal_path: str = TRADE_JOURNAL_PATH,
                cache_dir: str = CACHE_DIR) -> dict:
    """**部分减仓**落账(+25% 减半锁利 / 分批止盈的工具化;整仓清盘仍走 --sell)。

    补的缺口:此前部分卖出只能手工编辑 holdings.json——正是账本锁要防的最高危写路径。
    口径(与 journal 契约一致):journal 记**卖出那部分**为一笔完整平仓行(shares=本次
    卖出股数)。注意 trade_journal.stats 的 win_rate/expectancy 是**按笔等权**——部分
    减仓后"一行=一个完整仓位"不再成立,按笔读数会被短小的锁利腿美化,须看同函数的
    expectancy_w / win_rate_w(shares 加权,codex P1-3)。holdings 里该仓 shares 减去卖出量,
    **entry_date/cost/bucket/stop 全不动**(减仓不改变剩余仓的成本基础与纪律参数),
    cash += net。卖光(shares==持仓量)被拒绝——那是整仓清盘,须走 record_sell 以正确
    写 closed 段。护栏与 record_sell 同款:校验先行、失败零副作用、方向性净额、
    重复落账拒绝、交易日与 as_of 门禁、账本排他锁、原子写。
    """
    shares_out = _whole_shares(shares, "--shares")
    net = _positive(net, "--net")
    exit_px = _positive(exit_px, "--exit-px")
    pnl_pct = _finite(pnl_pct, "--pnl-pct")
    if reason_key not in REASONS:
        raise TradeRecordError(f"--reason 必须是 {sorted(REASONS)} 之一")
    with account_lock(holdings_path):
        holdings = _load_strict(holdings_path, "账户状态")
        journal = _load_strict(journal_path, "交易流水")
        _validate_holdings(holdings, holdings_path)
        if not isinstance(journal.get("trades"), list):
            raise TradeRecordError("journal.trades 结构非法——不落账")
        exit_date = _require_trading_day(_date8(date, "--date"), cache_dir, "--date")
        as_of = _date8(holdings.get("as_of"), "holdings.as_of")
        if exit_date < as_of:
            raise TradeRecordError(f"--date {exit_date} 早于账户 as_of {as_of}"
                                   "——账户 as_of 不可倒退,核对成交日")
        if exit_date != as_of and not advance_as_of:
            raise TradeRecordError(f"--date {exit_date} ≠ 账户 as_of {as_of}——先跑 "
                                   "holdings_confirm 推进账户日期再落账,或加 --advance-as-of "
                                   "在本次落账同一事务内推进(=确认该区间内无其他成交)")

        matches = [p for p in holdings["positions"] if p.get("ts_code") == ts_code]
        if len(matches) != 1:
            raise TradeRecordError(f"{ts_code} 在持仓中匹配到 {len(matches)} 条(需恰好 1 条)——不落账")
        position = matches[0]
        held = _whole_shares(position.get("shares"), "持仓 shares")
        if shares_out > held:
            raise TradeRecordError(f"减仓 {shares_out} 股 > 持仓 {held} 股——不落账")
        if shares_out == held:
            raise TradeRecordError(f"减仓股数等于全部持仓 {held}——整仓清盘请用 "
                                   "--sell(需写 closed 段),本命令只处理部分减仓;"
                                   "若股数抄错请先核对成交回报(codex P2-3)")
        remaining = held - shares_out
        # 零股卖出合法(送转后 1250 股卖 250):要求本次或剩余其一为整百,不把
        # 合法减仓推回手工编辑账本(codex P2-2)
        if shares_out % 100 and remaining % 100:
            raise TradeRecordError(f"--shares={shares_out} 与剩余 {remaining} 均非 100 倍数"
                                   "——A股卖出侧零股须一次性申报,核对成交回报")
        # mv 是派生字段,减仓后必须与 shares 同步重算;last 非正有限数则无法重算,
        # fail-loud——静默保留整仓旧 mv 会让 buy_list 的 account_value 虚高一个整仓、
        # 行业权重被稀释,可能放行本不该放行的买入(codex P1-1)
        has_mv = "mv" in position
        last_px = (_positive(position.get("last"), f"持仓 {ts_code} last(mv 需据其重算)")
                   if has_mv else None)
        # 减仓使既有 SELL 条件单股数大于剩余持仓 = 账本自相矛盾(挂单卖 1200 只剩 600),
        # 且条件单核验是 BUY 门禁的前置;fail-loud 要求先在券商改/撤单再落账(codex P1-2)
        # 权威格式是 {schema_version:2, orders:[...]}(account_state.validate_conditional_
        # orders_v2);此前只处理顶层裸 list,真实 structured_v2 会完全绕过守卫(codex P1)
        _co = holdings.get("conditional_orders")
        orders = _co.get("orders") if isinstance(_co, dict) else (_co if isinstance(_co, list) else None)
        if isinstance(orders, list):
            for o in orders:
                if not isinstance(o, dict) or o.get("ts_code") != ts_code:
                    continue
                if str(o.get("status", "active")).lower() != "active":
                    continue
                if str(o.get("side", "")).upper() != "SELL":
                    continue
                osh = o.get("shares")
                if isinstance(osh, int) and not isinstance(osh, bool) and osh > remaining:
                    raise TradeRecordError(
                        f"{ts_code} 存在 active SELL 条件单 {osh} 股 > 减仓后剩余 {remaining} 股"
                        "——先在券商改/撤该单并更新 holdings.conditional_orders 再落账")
        # 同日多腿(上午减半锁利 + 下午跌破止损)是真实合法序列,不能只按 (code,date)
        # 判重;但只按 (code,date,shares,px) 判重又会放过"同一笔重复录入、价格误敲一点"
        # (codex P1)。没有券商成交编号时,真正守住账务的是**股数守恒**而非键去重:
        #   ① 完全相同的 (股数,价) → 拒(几乎必然是重复提交);
        #   ② 当日已落账股数 + 本次 > 原始持仓 → 拒(超卖,这才是重复录入的真实危害);
        # 剩余的"同日同股数不同价"由 ② 兜住,超出部分一律拒绝。
        same_day_recorded = 0
        for t in journal["trades"]:
            if t.get("code") != ts_code or str(t.get("exit_date")) != exit_date:
                continue
            prior = t.get("shares")
            if isinstance(prior, int) and not isinstance(prior, bool) and prior > 0:
                same_day_recorded += prior
            if (prior == shares_out
                    and _finite(t.get("exit_px"), "journal exit_px") == exit_px):
                raise TradeRecordError(
                    f"journal 已有 {ts_code} exit_date={exit_date} {shares_out}股@{exit_px} "
                    "的同一笔流水——重复落账被拒绝")
        if same_day_recorded + shares_out > held + same_day_recorded:
            # held 已是扣减前腿后的余额,故等价于 shares_out > held(下方 ① 已校验);
            # 这里再显式挡一次"当日累计超过原始持仓"的情形,防止 held 读取路径变更时失守
            raise TradeRecordError(
                f"{ts_code} 当日已落账 {same_day_recorded} 股 + 本次 {shares_out} 股 "
                f"> 可减仓余额 {held} 股——疑似重复录入,核对成交回报")
        gross = shares_out * exit_px
        # 方向性护栏(同卖出腿):净回款只会被费用削减,不可能高于 gross
        if not (gross * (1 - NET_TOLERANCE) <= net <= gross):
            raise TradeRecordError(f"--net {net:,.2f} 应在 [{gross * (1 - NET_TOLERANCE):,.2f}, "
                                   f"{gross:,.2f}](股数×价−费)内——疑似抄错,核对成交回报")
        resolved_entry = entry_date or position.get("entry_date")
        if not resolved_entry:
            raise TradeRecordError("持仓无 entry_date 且未提供 --entry-date——journal 契约要求必填;"
                                   "历史仓请补 --entry-date(约数需加 --approx)")
        resolved_entry = _date8(resolved_entry, "entry_date")

        trade = _validate_journal_shape({
            "code": ts_code, "name": str(position.get("name") or ts_code),
            "bucket": _journal_bucket(position.get("bucket")),
            "entry_date": resolved_entry,
            "entry_px": _positive(position.get("cost"), f"持仓 {ts_code} cost"),
            "shares": shares_out, "exit_date": exit_date, "exit_px": exit_px,
            "pnl_pct": pnl_pct,
            "hold_days": _trading_hold_days(resolved_entry, exit_date, cache_dir),
            "reason": REASONS[reason_key], "approx": bool(approx),
        })
        new_positions = []
        for p in holdings["positions"]:
            if p.get("ts_code") == ts_code:
                # 只改 shares 与派生 mv;entry_date/cost/bucket/stop 一律不动
                updated = {**p, "shares": remaining}
                if last_px is not None:
                    updated["mv"] = round(remaining * last_px, 2)
                new_positions.append(updated)
            else:
                new_positions.append(p)
        new_journal = {**journal, "trades": [*journal["trades"], trade]}
        # 现金二位归整(真实资金账本;避免多次减仓累积二进制噪声,codex P2-7)
        new_holdings = {**holdings, "as_of": exit_date, "positions": new_positions,
                        "cash": round(float(holdings["cash"]) + net, 2)}
        # 预序列化两份 payload(allow_nan=False):序列化失败零副作用
        journal_text = json.dumps(new_journal, ensure_ascii=False, indent=2, allow_nan=False)
        holdings_text = json.dumps(new_holdings, ensure_ascii=False, indent=2, allow_nan=False)
        _atomic_write(journal_path, journal_text)
        try:
            _atomic_write(holdings_path, holdings_text)
        except BaseException:
            mv_hint = (f"、mv→{round(remaining * last_px, 2)}" if last_px is not None else "")
            print(f"!!! journal 已写入但 holdings 替换失败——中间态!恢复指引:"
                  f"删除 {journal_path} 中 code={ts_code} exit_date={exit_date} 的最后一行,"
                  f"或手工完成 holdings 修改({ts_code} shares→{remaining}{mv_hint},"
                  f"cash+={net})后勿重跑")
            raise
    return {"side": "TRIM", "ts_code": ts_code, "shares": shares_out,
            "remaining": remaining, "net": net, "pnl_pct": pnl_pct,
            "cash": new_holdings["cash"], "positions": len(new_holdings["positions"])}


def _snapshot_row(ts_code: str, as_of: str, holdscore_dir: str) -> dict:
    path = Path(holdscore_dir) / f"{as_of}_factor.json"
    if not path.exists():
        raise TradeRecordError(f"无 factor snapshot {path}——买入需要机器数据补齐行业/现价")
    rows = json.load(path.open(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(rows, list):
        raise TradeRecordError(f"{path} 顶层必须是列表——不落账")
    matches = [r for r in rows if isinstance(r, dict) and str(r.get("ts_code")) == ts_code]
    if len(matches) != 1:
        raise TradeRecordError(f"{ts_code} 在 {as_of} snapshot 匹配到 {len(matches)} 条"
                               "(需恰好 1 条)——代码敲错或不在宇宙,不落账")
    row = matches[0]
    name, industry = row.get("name"), row.get("industry")
    if not (isinstance(name, str) and name.strip()) or not (isinstance(industry, str) and industry.strip()):
        raise TradeRecordError(f"{ts_code} snapshot 行 name/industry 缺失——不落账")
    last = row.get("last")
    if isinstance(last, bool) or not isinstance(last, (int, float)) \
            or not math.isfinite(float(last)) or float(last) <= 0:
        raise TradeRecordError(f"{ts_code} snapshot last={last!r} 非正有限数(停牌/无价)——不落账")
    return {"name": name, "industry": industry, "last": float(last)}


def record_buy(ts_code: str, *, date: str, shares: int, cost_px: float, net: float,
               bucket: str = "long", advance_as_of: bool = False,
               holdings_path: str = HOLDINGS_PATH,
               holdscore_dir: str = HOLDSCORE_DIR,
               cache_dir: str = CACHE_DIR) -> dict:
    """新建仓落账(加仓不支持)。校验先行,失败零副作用。"""
    shares = _whole_shares(shares, "--shares")
    if shares % 100:
        raise TradeRecordError(f"--shares 必须是 100 的倍数,得到 {shares}")
    cost_px = _positive(cost_px, "--cost-px")
    net = _positive(net, "--net")
    if bucket not in BUCKET_TO_JOURNAL:
        raise TradeRecordError(f"--bucket 必须是 {sorted(BUCKET_TO_JOURNAL)} 之一")
    with account_lock(holdings_path):
        holdings = _load_strict(holdings_path, "账户状态")
        _validate_holdings(holdings, holdings_path)
        buy_date = _require_trading_day(_date8(date, "--date"), cache_dir, "--date")
        as_of = _date8(holdings.get("as_of"), "holdings.as_of")
        if buy_date < as_of:
            raise TradeRecordError(f"--date {buy_date} 早于账户 as_of {as_of}"
                                   "——账户 as_of 不可倒退,核对成交日")
        if buy_date != as_of and not advance_as_of:
            raise TradeRecordError(f"--date {buy_date} ≠ 账户 as_of {as_of}——先跑 "
                                   "holdings_confirm 推进账户日期再落账,或加 --advance-as-of "
                                   "在本次落账同一事务内推进(=确认该区间内无其他成交)")
        if any(p.get("ts_code") == ts_code for p in holdings["positions"]):
            raise TradeRecordError(f"{ts_code} 已在持仓——加仓请手工编辑,不落账")
        gross = shares * cost_px
        # 方向性护栏:买入实扣只会被费用抬高,不可能低于 gross
        if not (gross <= net <= gross * (1 + NET_TOLERANCE)):
            raise TradeRecordError(f"--net {net:,.2f} 应在 [{gross:,.2f}, "
                                   f"{gross * (1 + NET_TOLERANCE):,.2f}](股数×价+费)内——疑似抄错")
        cash = float(holdings["cash"])
        if net > cash:
            raise TradeRecordError(f"实扣 {net:,.2f} 超过现金 {cash:,.2f}——账对不上,不落账")

        row = _snapshot_row(ts_code, buy_date, holdscore_dir)
        new_holdings = {**holdings,
                        "as_of": buy_date,  # 前进式推进(--advance-as-of 时)或维持;与落账同一原子写
                        "positions": [*holdings["positions"], {
                            "ts_code": ts_code, "name": row["name"],
                            "industry": row["industry"], "shares": shares,
                            "cost": cost_px, "last": row["last"],
                            "mv": round(shares * row["last"], 2),
                            # bucket 落**中文**(与 journal 三档契约、既有账本一致):
                            # 写英文会让 account_state 的短线席位与 intraday 的 +25%
                            # 止盈线静默失效(跨层审计 P1)
                            "stop": None, "bucket": BUCKET_TO_JOURNAL[bucket],
                            "entry_date": buy_date,
                            "bucket_note": f"{buy_date} trade_record 落账",
                            "tag": "", "theme": "", "watch": False,
                        }],
                        "cash": round(cash - net, 2)}   # 与 SELL/TRIM 同按分归整(codex P2)
        holdings_text = json.dumps(new_holdings, ensure_ascii=False, indent=2, allow_nan=False)
        _atomic_write(holdings_path, holdings_text)
    return {"side": "BUY", "ts_code": ts_code, "shares": shares, "net": net,
            "cash": new_holdings["cash"], "positions": len(new_holdings["positions"])}


def main(argv: "list[str] | None" = None) -> None:
    ap = argparse.ArgumentParser(description="人工成交落账(用户本人运行;智能体不得调用)")
    side = ap.add_mutually_exclusive_group(required=True)
    side.add_argument("--sell", metavar="TS_CODE", help="整仓卖出的 ts_code")
    side.add_argument("--buy", metavar="TS_CODE", help="新建仓的 ts_code")
    side.add_argument("--trim", metavar="TS_CODE",
                      help="部分减仓的 ts_code(+25%% 减半锁利/分批止盈;整仓清盘用 --sell)")
    ap.add_argument("--date", required=True, help="成交日 YYYYMMDD(须=账户 as_of 且为交易日)")
    ap.add_argument("--net", type=float, required=True,
                    help="券商回报净额:卖=净回款,买=实扣(含费)")
    ap.add_argument("--exit-px", type=float, help="卖出成交均价(--sell/--trim 必填)")
    ap.add_argument("--pnl-pct", type=float,
                    help="账户口径盈亏%%,抄券商 App(--sell/--trim 必填;不自行推导。"
                         "--trim 填**本次卖出部分**的已实现盈亏%%,不是整仓浮盈)")
    ap.add_argument("--cost-px", type=float, help="买入成交均价(--buy 必填)")
    ap.add_argument("--shares", type=int,
                    help="股数(--buy/--trim 必填,100 的倍数;--trim 为本次卖出股数)")
    ap.add_argument("--reason", choices=sorted(REASONS), help="卖出理由(--sell 必填)")
    ap.add_argument("--entry-date", default=None,
                    help="卖出腿原始买入日(持仓行有 entry_date 可省;历史仓必填)")
    ap.add_argument("--approx", action="store_true",
                    help="entry 信息为历史回忆约数时显式声明(缺省 False)")
    ap.add_argument("--bucket", default="long", choices=sorted(BUCKET_TO_JOURNAL),
                    help="买入仓别(缺省 long=长线)")
    ap.add_argument("--advance-as-of", action="store_true",
                    help="成交日晚于账户 as_of 时,在本次落账同一事务内前进式推进 as_of"
                         "(=人工确认该区间内无其他成交,免去先单独跑 holdings_confirm)")
    a = ap.parse_args(argv)

    if a.sell:
        if a.exit_px is None or a.reason is None or a.pnl_pct is None:
            raise TradeRecordError("--sell 需要 --exit-px、--pnl-pct 与 --reason")
        result = record_sell(a.sell, date=a.date, net=a.net, exit_px=a.exit_px,
                             pnl_pct=a.pnl_pct, reason_key=a.reason,
                             entry_date=a.entry_date, approx=a.approx,
                             advance_as_of=a.advance_as_of)
        print(f"SELL {result['ts_code']} {result['shares']}股 净回款 {result['net']:,.2f}"
              f" 盈亏 {result['pnl_pct']:+.2f}%(券商口径)")
    elif a.trim:
        if a.exit_px is None or a.reason is None or a.pnl_pct is None or a.shares is None:
            raise TradeRecordError("--trim 需要 --shares、--exit-px、--pnl-pct 与 --reason")
        result = record_trim(a.trim, date=a.date, shares=a.shares, net=a.net,
                             exit_px=a.exit_px, pnl_pct=a.pnl_pct, reason_key=a.reason,
                             entry_date=a.entry_date, approx=a.approx,
                             advance_as_of=a.advance_as_of)
        print(f"TRIM {result['ts_code']} 卖出{result['shares']}股 剩余{result['remaining']}股"
              f" 净回款 {result['net']:,.2f} 盈亏 {result['pnl_pct']:+.2f}%(券商口径)")
        print("  注:剩余仓 last/mv 仍是落账前的价格口径(本工具不刷行情);"
              "EOD 估值以 holdings_watch 为准")
    else:
        if a.cost_px is None or a.shares is None:
            raise TradeRecordError("--buy 需要 --cost-px 与 --shares")
        result = record_buy(a.buy, date=a.date, shares=a.shares, cost_px=a.cost_px,
                            net=a.net, bucket=a.bucket, advance_as_of=a.advance_as_of)
        print(f"BUY {result['ts_code']} {result['shares']}股 实扣 {result['net']:,.2f}")
    print(f"→ 现金 {result['cash']:,.2f} | 持仓 {result['positions']} 只"
          f"(请核对与券商 App 一致;不一致立即手工修正)")


if __name__ == "__main__":
    main()
