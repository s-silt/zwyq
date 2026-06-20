"""TDD:幸存者清单选择逻辑 scripts.survivors.pick_survivors(纯函数)。

只挑达标质地档的 record,按 tier 优先级(🟢→🟡)+ entry 分降序;缺 entry 分排末。
"""
from __future__ import annotations

from scripts.survivors import pick_survivors


def _r(grade: str, score: float | None) -> dict:
    return {"ts_code": f"X{grade}{score}", "tier": {"grade": grade}, "entry": {"score": score}}


def test_pick_default_only_green_sorted_by_score_desc() -> None:
    recs = [_r("🟢", 70.0), _r("🟡", 99.0), _r("🟢", 95.0), _r("⛔", 50.0), _r("🔴", 80.0)]
    surv = pick_survivors(recs)
    assert [r["tier"]["grade"] for r in surv] == ["🟢", "🟢"]  # 只 🟢
    assert [r["entry"]["score"] for r in surv] == [95.0, 70.0]  # 分降序


def test_pick_can_include_yellow_green_first() -> None:
    recs = [_r("🟡", 88.0), _r("🟢", 60.0), _r("🔴", 90.0)]
    surv = pick_survivors(recs, grades=("🟢", "🟡"))
    assert [r["tier"]["grade"] for r in surv] == ["🟢", "🟡"]  # 🟢 优先于 🟡


def test_pick_none_score_sorts_last() -> None:
    recs = [_r("🟢", None), _r("🟢", 40.0)]
    surv = pick_survivors(recs)
    assert [r["entry"]["score"] for r in surv] == [40.0, None]  # 缺分排末


def test_pick_empty_when_no_match() -> None:
    assert pick_survivors([_r("🔴", 90.0), _r("⛔", 80.0)]) == []
