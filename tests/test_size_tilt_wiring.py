"""X-08 生产接线(用户批准 2026-08-08):BUY 候选排序优先 D10 小市值桶。

依据:增量(S−PROD)净 +0.601%/期 NW t3.07(贴线,含 ~1/3 β)。语义=选股偏好,
四关/治理核实/行业上限/资金分配全部不变;桶在 D10 全档内划(ex-ante,X-08 同口径)。
"""
from __future__ import annotations

import pytest

POLICY = {"policy_version": "2", "target_positions": 10, "target_weight": 0.10,
          "industry_cap": 0.20, "lot_size": 100, "min_cash": 0}


def _row(ts: str, mv: float, decile: int = 10) -> dict:
    return {"ts_code": ts, "name": ts, "decile": decile, "mv": mv}


def _a(ts: str, score: float, size_rank: "int | None" = None,
       bucket: "str | None" = None, industry: str = "化工原料") -> dict:
    a = {"ts_code": ts, "name": ts, "industry": industry, "score": score, "last": 10.0,
         "eligible_buy": True, "reason_codes": ["D10", "TIER_GREEN", "FACTCHECK_CLEAR"],
         "governance_red": False}
    if size_rank is not None:
        a["size_rank"], a["size_bucket"] = size_rank, bucket
    return a


def test_size_tercile_ranks_splits_d10_by_mv():
    from scripts.buy_list import size_tercile_ranks

    rows = [_row(f"60000{i}.SH", mv=float(i * 10)) for i in range(1, 7)]
    got = size_tercile_ranks(rows)
    assert got["600001.SH"] == (0, "小") and got["600002.SH"] == (0, "小")
    assert got["600003.SH"] == (1, "中") and got["600004.SH"] == (1, "中")
    assert got["600005.SH"] == (2, "大") and got["600006.SH"] == (2, "大")


def test_size_tercile_ranks_ignores_non_d10():
    from scripts.buy_list import size_tercile_ranks

    rows = ([_row(f"60000{i}.SH", mv=float(i * 10)) for i in range(1, 7)]
            + [_row("600099.SH", mv=1.0, decile=9)])   # 更小但不在 D10
    got = size_tercile_ranks(rows)
    assert "600099.SH" not in got
    assert got["600001.SH"][0] == 0                    # 桶界不被非 D10 行影响


def test_size_tercile_ranks_fail_loud_on_bad_mv():
    from scripts.buy_list import size_tercile_ranks

    rows = [_row(f"60000{i}.SH", mv=float(i * 10)) for i in range(1, 6)]
    rows.append({"ts_code": "600006.SH", "name": "600006.SH", "decile": 10, "mv": None})
    with pytest.raises(SystemExit):                    # 缺市值=排序静默失效,必须炸
        size_tercile_ranks(rows)


def test_size_tercile_ranks_tiny_panel_degrades_to_neutral():
    """<3 行 D10=无三分位语义(X-04 mv_terciles 同精神)→ 空映射,排序回退 score;
    这是合法退化截面,不是数据损坏,不 fail-loud。"""
    from scripts.buy_list import size_tercile_ranks

    assert size_tercile_ranks([_row("600001.SH", 10.0), _row("600002.SH", 20.0)]) == {}


def test_buy_slot_contention_small_bucket_wins_over_score():
    from ashare_gauntlet.portfolio_decision import decide_states

    held = {f"60010{i}.SH": {"ts_code": f"60010{i}.SH", "name": "x", "industry": f"行业{i}",
                             "mv": 10_000.0, "last": 10.0} for i in range(1, 10)}
    cands = [_a("600001.SH", score=0.99, size_rank=2, bucket="大", industry="行业A"),
             _a("600002.SH", score=0.50, size_rank=0, bucket="小", industry="行业B")]
    ds = decide_states(cands, held, POLICY, account_value=100_000.0, cash=90_000.0)
    d = {x["ts_code"]: x for x in ds}
    # 9 持仓 + 1 空位:小桶低分候选拿走唯一席位,大桶高分让位
    assert d["600002.SH"]["state"] == "BUY"
    assert d["600001.SH"]["state"] == "WAIT"
    assert "PORTFOLIO_FULL" in d["600001.SH"]["reason_codes"]


def test_display_order_small_bucket_first_and_evidence_carries_bucket():
    from ashare_gauntlet.portfolio_decision import decide_states

    cands = [_a("600001.SH", score=0.99, size_rank=2, bucket="大", industry="行业A"),
             _a("600002.SH", score=0.50, size_rank=0, bucket="小", industry="行业B")]
    ds = decide_states(cands, {}, POLICY, account_value=100_000.0, cash=90_000.0)
    assert [d["ts_code"] for d in ds] == ["600002.SH", "600001.SH"]
    assert ds[0]["evidence"]["size_bucket"] == "小"
    assert ds[0]["evidence"]["size_rank"] == 0


def test_without_size_rank_falls_back_to_score_order():
    from ashare_gauntlet.portfolio_decision import decide_states

    cands = [_a("600002.SH", score=0.50), _a("600001.SH", score=0.99)]
    ds = decide_states(cands, {}, POLICY, account_value=100_000.0, cash=90_000.0)
    assert [d["ts_code"] for d in ds] == ["600001.SH", "600002.SH"]   # 旧语义不变
