"""股息率呈现叠加(展示层,**绝不进 composite**):从 daily_basic 缓存读 dv_ttm。

边界(严守 CLAUDE.md 研究不变量):
- 生产 composite 固定 EP+BP+IVOL(负向);dv_ttm 只做**展示叠加**,不入分、不改机器
  状态、不把 WAIT 提升为 BUY。是否把股息做成因子须另走五门 gauntlet(独立实验)。
- dv_ttm=近 12 个月滚动股息率(%),dv_ratio=股息率(%),取自 as-of 当日 daily_basic
  分区里**已可见**的滚动指标——是历史滚动指示,不是未来分红预测、不是承诺收益。
- 缺失一律标 None(MISSING),**绝不当 0**;分区缺失 fail-loud(不得解释为"无分红")。

本模块 import 阶段无任何 I/O 副作用。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from ashare_gauntlet.config import CACHE_DIR

# daily_basic 里的股息列(个别退化分区可能只带其中之一,读时按存在列取)
DIVIDEND_COLUMNS = ("dv_ttm", "dv_ratio")


class DividendDataUnavailable(FileNotFoundError):
    """daily_basic 当日分区缺失——股息叠加不可用(调用方须当'辅助数据不可用',不当无分红)。"""


class DividendDataDegraded(DividendDataUnavailable):
    """分区存在但股息列**全市场整列 NULL**——上游字段退化,不得解释为'全市场无分红'。

    判据放在全市场范围而非本次查询的少数 code:个别 code 无分红是正常 None,
    整列 NULL 才是退化信号。"""


def _partition_path(as_of: str, cache_dir: str = CACHE_DIR) -> Path:
    return Path(cache_dir) / "daily_basic" / f"{as_of}.parquet"


def _finite_or_none(value: object) -> float | None:
    """把值收敛为有限 float 或 None(NaN/Inf/非数皆 None,不伪造 0)。"""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def dividend_yields(
    codes: Iterable[str],
    as_of: str,
    *,
    cache_dir: str = CACHE_DIR,
) -> dict[str, dict[str, float | None]]:
    """返回 {ts_code: {"dv_ttm": float|None, "dv_ratio": float|None}}(展示用)。

    - 分区缺失 → 抛 DividendDataUnavailable(fail-loud;调用方标 UNAVAILABLE,不当无分红)。
    - 分区存在但 dv_ttm 缺列 / 全市场整列 NULL → 抛 DividendDataDegraded(fail-loud;
      上游字段退化,调用方标 DEGRADED,不当无分红)。dv_ratio 单独退化不算(不被消费)。
    - 某 code 无行 / 值为 NaN / 缺列 → 该字段 None(MISSING),不填 0。
    - codes 去重且保持首次出现顺序;只查所给 code,不改动任何机器状态。
    """
    path = _partition_path(as_of, cache_dir)
    if not path.exists():
        raise DividendDataUnavailable(
            f"daily_basic 分区缺失: {path}——股息叠加不可用(勿解释为无分红)")

    wanted = list(dict.fromkeys(str(code) for code in codes))
    out: dict[str, dict[str, float | None]] = {
        code: {col: None for col in DIVIDEND_COLUMNS} for code in wanted
    }
    if not wanted:
        return out

    frame = pd.read_parquet(path)
    if "ts_code" not in frame.columns:
        raise DividendDataUnavailable(
            f"daily_basic 分区缺 ts_code 列(退化 schema): {path}")
    present_cols = [col for col in DIVIDEND_COLUMNS if col in frame.columns]
    if not present_cols:
        # 退化分区连一列股息都没有:显式不可用,不静默返回全 0/全 None 冒充有数据
        raise DividendDataUnavailable(
            f"daily_basic 分区不含股息列 {DIVIDEND_COLUMNS}(退化 schema): {path}")
    # 退化判据盯 dv_ttm(展示层唯一消费字段;dv_ratio 单独退化不影响叠加可用),
    # 且取全市场范围而非本次查询的少数 code:个别 code 无分红是正常 None,
    # 整列 NULL(或分区空表)才是上游字段退化
    ttm_missing = "dv_ttm" not in frame.columns
    if ttm_missing or bool(frame["dv_ttm"].isna().all()):
        detail = "缺列" if ttm_missing else "整列 NULL"
        raise DividendDataDegraded(
            f"daily_basic/{as_of} 的 dv_ttm {detail}"
            f"——上游字段退化,股息叠加不可用(勿解释为无分红): {path}")

    indexed = frame.set_index("ts_code")
    for code in wanted:
        if code not in indexed.index:
            continue
        row = indexed.loc[code]
        if isinstance(row, pd.DataFrame):  # 理论上 ts_code 唯一;重复时保守取首行
            row = row.iloc[0]
        for col in present_cols:
            out[code][col] = _finite_or_none(row.get(col))
    return out


def indicative_ttm_cash(
    dv_ttm_pct: float | None,
    market_value: float | None,
) -> float | None:
    """展示用 TTM 指示性现金流 = 市值 × dv_ttm/100(历史滚动指示,非预测、非到账)。

    任一输入缺失/非有限 → None(不当 0)。绝不写入实际收益;实际到账须券商流水人工确认。
    """
    dv = _finite_or_none(dv_ttm_pct)
    mv = _finite_or_none(market_value)
    if dv is None or mv is None:
        return None
    return round(mv * dv / 100.0, 2)
