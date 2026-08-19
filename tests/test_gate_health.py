"""门禁证据体检:五门复用既有判据,退化必须报,口径不得混用。"""
from __future__ import annotations

import math

import pytest

from ashare_gauntlet import gate_health as gh


def _rows(ic=0.05, n=60, factor="EP"):
    """构造能过五门的合成读数(IC 稳定为正、腿收益为正、成本极小)。"""
    out = []
    for i in range(n):
        year = 2020 + i // 12
        out.append({
            "date": f"{year}{(i % 12) + 1:02d}15",
            "mkt_fwd": 0.01 if i % 2 else -0.01,
            "cost_rt": 0.0001,
            f"IC_{factor}": ic,
            f"SPR_{factor}": 0.02,
            f"QHI_{factor}": 0.03, f"QLO_{factor}": -0.01,
            f"TO_{factor}": 0.1, f"TOHI_{factor}": 0.1, f"TOLO_{factor}": 0.1,
        })
    return out


def test_gate_status_passes_for_strong_factor():
    r = gh.factor_gate_status(_rows(), "EP")
    assert r["status"] == "PASS" and r["passed"] is True and r["reasons"] == []
    assert r["nw_t"] > 3.0


def test_gate_status_fails_when_evidence_degrades():
    # IC 接近 0 → NW t 过不了 T_ADMIT=3.0 → 必须 FAIL(不是静默通过)
    r = gh.factor_gate_status(_rows(ic=0.0005), "EP")
    assert r["status"] == "FAIL" and r["passed"] is False
    assert any("NW t>3" in x for x in r["reasons"])


def test_missing_column_is_not_treated_as_pass():
    """读数缺列 → NO_DATA,绝不当作"证据仍成立"(未知不解释为安全)。"""
    r = gh.factor_gate_status([{"date": "20260101", "mkt_fwd": 0.01}], "EP")
    assert r["status"] == "NO_DATA" and r["passed"] is None


def test_composite_caliber_is_gross_excess():
    """nw_t 必须是**毛超额**口径(与 composite_backtest 报告/methodology §10 同),
    否则基线比对会拿两个口径做差,凭空造出"退化"或掩盖真退化。"""
    rows = [{"ret_PROD": 0.02, "mkt_fwd": 0.01, "TO_PROD": 0.4, "cost_rt": 0.004}
            for _ in range(50)]
    c = gh.composite_nw_t(rows)
    assert c["caliber"] == "gross_excess"
    assert c["gross_mean_pct"] == pytest.approx(1.0)        # (0.02-0.01)*100
    assert c["net_mean_pct"] == pytest.approx(0.84)         # 再减 0.4*0.004
    assert c["nw_t"] != c["nw_t_net"]                       # 两个口径必须是不同的数


def test_compare_flags_degradation_and_drift():
    strong = gh.build_report(_rows(ic=0.05), None)
    weak = gh.build_report(_rows(ic=0.0005), None)
    # 由 PASS 变 FAIL → DEGRADED(合成读数只含 EP 列,BP/IVOL 恒 NO_DATA,故只看 EP)
    found = gh.compare_to_baseline(weak, strong)
    assert any(f["target"] == "EP" and f["level"] == "DEGRADED"
               and f["issue"] == "GATE_FAIL" for f in found)
    # 自己跟自己比:EP 不应有任何发现(BP/IVOL 的 NO_DATA 是构造缺列所致,属预期)
    same = gh.compare_to_baseline(strong, strong)
    assert [f for f in same if f["target"] == "EP"] == []


def test_composite_below_fold_is_degraded():
    cur = {"factors": [], "composite": {"status": "OK", "nw_t": 1.5}}
    base = {"factors": [], "composite": {"status": "OK", "nw_t": 3.1}}
    found = gh.compare_to_baseline(cur, base)
    assert any(f["issue"] == "T_BELOW_FOLD" and f["level"] == "DEGRADED" for f in found)


def test_no_data_composite_never_reported_healthy():
    cur = {"factors": [], "composite": {"status": "NO_DATA", "nw_t": None, "note": "缺列"}}
    found = gh.compare_to_baseline(cur, {"factors": [], "composite": {"nw_t": 3.0}})
    assert any(f["level"] == "DEGRADED" and f["target"] == "composite" for f in found)


def test_build_report_shape():
    rep = gh.build_report(_rows(), None, as_of="20260819")
    assert rep["schema_version"] == "gate_health.v1"
    assert [f["factor"] for f in rep["factors"]] == list(gh.PRODUCTION_FACTORS)
    assert rep["sample"]["n"] == 60
    # EP 有数据、BP/IVOL 无 → all_gates_pass 必须 False(缺数据不算通过)
    assert rep["all_gates_pass"] is False
    assert not math.isnan(rep["factors"][0]["nw_t"])


def test_negative_factor_degradation_is_detected():
    """codex P1:IVOL 等负向因子 t 为负,直接做差会把 -17→-10 的退化算成"上升"而漏报。"""
    base = {"factors": [{"factor": "IVOL", "status": "PASS", "nw_t": -17.0}], "composite": {}}
    worse = {"factors": [{"factor": "IVOL", "status": "PASS", "nw_t": -10.0}], "composite": {}}
    found = gh.compare_to_baseline(worse, base)
    assert any(f["issue"] == "T_DROP" and f["target"] == "IVOL" for f in found)
    # |t| 上升(更显著)不该报
    assert gh.compare_to_baseline(
        {"factors": [{"factor": "IVOL", "status": "PASS", "nw_t": -20.0}], "composite": {}},
        base) == []


def test_sign_flip_outranks_drift():
    """方向翻转=因子语义变了,必须 DEGRADED,不能被降级成 DRIFT 观察项。"""
    base = {"factors": [{"factor": "IVOL", "status": "PASS", "nw_t": -17.0}], "composite": {}}
    flip = {"factors": [{"factor": "IVOL", "status": "PASS", "nw_t": 5.0}], "composite": {}}
    found = gh.compare_to_baseline(flip, base)
    assert [f["issue"] for f in found] == ["T_SIGN_FLIP"]
    assert found[0]["level"] == "DEGRADED"
