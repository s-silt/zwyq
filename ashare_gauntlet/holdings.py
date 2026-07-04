"""持仓盯盘估值 —— 纯函数层(只算不判)。

设计边界(见 memory scoring-needs-theory / model-aware-judgment):本模块**只输出确定性
数字**(盈亏/距止损/前复权MA20/距20低/持仓交易日数),**不做任何信号判定**——"要不要
提醒/怎么解读"留给盯盘时的 Claude 按 bucket 规则 + 生意模式判断,不把阈值写死成 magic
number。

口径钉死:
- close / 盈亏 / 距止损:**不复权**(与 cost/stop 同为实际成交价口径);
- MA20 / 距20日低:**前复权**(归一到当日因子,除权日不跳变);
- 距止损 dist_stop = (close−stop)/close(从现价再跌多少%到止损)。

fail-loud 分层(见 memory analysis-priorities):单只/单字段坏数据→降级标注继续(盯盘5只
不因1只瘫痪);缓存整体不是今日→硬失败(拒绝拿旧价当今日)。
"""
from __future__ import annotations

from collections.abc import Sequence


class StaleCacheError(RuntimeError):
    """本地缓存最新日 ≠ 传入的今日 —— backfill 未跑成功,所有价都是旧的;
    静默用旧价盯盘会误判(昨天没破 stop 不代表今天没破),fail-loud。"""


def pct_change(value: float, ref: float) -> float:
    """相对参照的百分比变化 = (value−ref)/ref×100(盈亏/距MA20/距20低共用)。

    ref 是价格类参照(cost/ma/low),不可能 ≤0;≤0 只能是坏数据 → fail-loud。
    """
    if ref <= 0:
        raise ValueError(f"参照值须为正(价格类),得到 {ref}")
    return (value - ref) / ref * 100.0


def downside_to_stop(close: float, stop: float) -> float:
    """从现价再跌多少%到止损 = (close−stop)/close×100(分母=现价,盯盘最直观)。

    正=止损在下方尚有缓冲,负=已跌破止损(不掩盖)。close≤0 → fail-loud。
    """
    if close <= 0:
        raise ValueError(f"现价须为正,得到 {close}")
    return (close - stop) / close * 100.0


def qfq_series(raw: Sequence[float], adj: Sequence[float], latest_adj: float) -> list[float]:
    """前复权归一:逐点 raw×adj/latest_adj(把历史价折算到当日复权因子口径)。

    raw 与 adj 一一对应,长度须一致;latest_adj>0。不匹配/非正 → fail-loud。
    """
    if len(raw) != len(adj):
        raise ValueError(f"价与复权因子长度不一致:{len(raw)} vs {len(adj)}")
    if latest_adj <= 0:
        raise ValueError(f"当日复权因子须为正,得到 {latest_adj}")
    return [r * a / latest_adj for r, a in zip(raw, adj)]


def moving_average(values: Sequence[float], window: int) -> float | None:
    """最近 window 个值的均线;不足 window → None(降级,不缩窗伪造)。"""
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def min_low(values: Sequence[float], window: int) -> float | None:
    """最近 window 个值的最小值(近 window 日低);不足 → None(降级)。"""
    if len(values) < window:
        return None
    return min(values[-window:])


def held_trading_days(entry_date: str, as_of: str, trade_days: Sequence[str]) -> int:
    """entry→as_of 的交易日数(含两端;短线 10 交易日时间止损用)。

    trade_days 为 YYYYMMDD 字符串(可直接字典序比较);entry 当天算第 1 日。
    """
    return sum(1 for d in trade_days if entry_date <= d <= as_of)


def verify_as_of(cache_latest: str, expected: str) -> None:
    """校验缓存最新日 == 今日;不等 → StaleCacheError(系统性 fail-loud)。"""
    if cache_latest != expected:
        raise StaleCacheError(
            f"缓存最新日 {cache_latest} ≠ 今日 {expected} —— backfill 未刷新,"
            f"拒绝拿旧价盯盘")


def is_date_partition(basename: str) -> bool:
    """True 当 basename 是日期分区缓存文件 YYYYMMDD.parquet(恰 8 位数字)。

    daily/adj_factor 缓存目录里混有 per-symbol 区间文件
    (<ts_code>_<start>_<end>.parquet),它们不是交易日;不滤掉则
    sorted(glob)[-1] 会取到 ts_code 命名文件,把 as_of 污染成代码串(实测踩过)。
    """
    suffix = ".parquet"
    if not basename.endswith(suffix):
        return False
    stem = basename[: -len(suffix)]
    return len(stem) == 8 and stem.isdigit()


def build_position_record(
    pos: dict,
    close: float | None,
    pct_chg: float | None,
    qfq_closes: Sequence[float],
    qfq_lows: Sequence[float],
    as_of: str,
    trade_days: Sequence[str],
    window: int = 20,
) -> dict:
    """组装一只持仓的估值 record(只算不判)。

    透传判断必需字段(bucket/bucket_note/theme/cost/stop…)+ 算出的数字。
    close=None(无当日行情)→ error 标注、数值字段 None,其余持仓照算(单只降级);
    qfq 历史不足 window → ma20/距离字段 None(单字段降级),盈亏等不依赖历史的照算。
    pct 类 round(2) 便于 JSON 展示(round 是展示、不是判断)。
    """
    ident = {
        "ts_code": pos["ts_code"],
        "name": pos["name"],
        "theme": pos.get("theme"),
        "bucket": pos.get("bucket"),
        "bucket_note": pos.get("bucket_note"),
        "shares": pos.get("shares"),
        "cost": pos.get("cost"),
        "stop": pos.get("stop"),
        "entry_date": pos.get("entry_date"),
    }
    null_fields = {
        "close": None, "pct_chg": None, "pnl_pct": None, "dist_stop_pct": None,
        "ma20": None, "dist_ma20_pct": None, "dist_low20_pct": None, "held_days": None,
    }

    if close is None:
        return {**ident, **null_fields, "error": "无当日行情(停牌/代码错)"}

    cost = pos["cost"]
    stop = pos["stop"]
    entry_date = pos.get("entry_date")

    ma20 = moving_average(qfq_closes, window)
    low20 = min_low(qfq_lows, window)
    last_qfq = qfq_closes[-1] if qfq_closes else None
    dist_ma20 = round(pct_change(last_qfq, ma20), 2) if (ma20 is not None and last_qfq is not None) else None
    dist_low20 = round(pct_change(last_qfq, low20), 2) if (low20 is not None and last_qfq is not None) else None
    held = held_trading_days(entry_date, as_of, trade_days) if entry_date else None

    return {
        **ident,
        "close": close,
        "pct_chg": pct_chg,
        "pnl_pct": round(pct_change(close, cost), 2),
        "dist_stop_pct": round(downside_to_stop(close, stop), 2),
        "ma20": round(ma20, 2) if ma20 is not None else None,
        "dist_ma20_pct": dist_ma20,
        "dist_low20_pct": dist_low20,
        "held_days": held,
        "error": None,
    }
