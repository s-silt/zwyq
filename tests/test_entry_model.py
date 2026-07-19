"""entry_model:入场特征纯函数 + 八条门禁真值表(spec §6)。"""
from __future__ import annotations

import pandas as pd
import pytest


def _panel(vals: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(vals)


def test_dma20_and_insufficient_history():
    from ashare_gauntlet.entry_model import dma20

    flat = _panel({"A": [10.0] * 20, "B": [10.0] * 19 + [12.0]})
    out = dma20(flat)
    assert out["A"] == pytest.approx(0.0)
    assert out["B"] == pytest.approx(12.0 / 10.1 - 1.0)   # MA20 含最新根
    assert pd.isna(dma20(_panel({"A": [10.0] * 19}))["A"])


def test_ret_n_and_gap():
    from ashare_gauntlet.entry_model import gap_pct, ret_n

    p = _panel({"A": [10, 10, 10, 10, 10, 11.0]})
    assert ret_n(p, 5)["A"] == pytest.approx(0.10)
    g = gap_pct(pd.Series({"A": 10.3}), pd.Series({"A": 10.0}))
    assert g["A"] == pytest.approx(0.03)


def _stats(**kw) -> dict:
    base = {"prior_registered": True, "net_diff": 0.10, "loyo_signs": [1, 1, 1],
            "up_diff": 0.05, "dn_diff": 0.15, "sig_t": 2.5, "coverage": 0.5,
            "mdd_rule": -0.20, "mdd_base": -0.25, "neighborhood_diffs": [0.08, 0.10, 0.12]}
    base.update(kw)
    return base


def test_gate_verdict_all_pass():
    from ashare_gauntlet.entry_model import gate_verdict

    v = gate_verdict(_stats())
    assert v["passed"] is True and v["failed"] == []


def test_gate_verdict_truth_table():
    from ashare_gauntlet.entry_model import gate_verdict

    cases = [
        (_stats(prior_registered=False), "PRIOR_REGISTERED"),
        (_stats(net_diff=-0.01), "NET_POSITIVE_VS_BASE"),
        (_stats(loyo_signs=[1, -1, 1]), "LOYO_STABLE"),
        (_stats(up_diff=-0.05, dn_diff=0.15), "REGIME_CONSISTENT"),
        (_stats(sig_t=1.5), "SIGNIFICANT_AT_PRESPECIFIED_HORIZON"),
        (_stats(coverage=0.10), "COVERAGE_FLOOR"),
        (_stats(mdd_rule=-0.40, mdd_base=-0.25), "MDD_NOT_WORSE"),
        (_stats(neighborhood_diffs=[0.08, -0.01, 0.12]), "NEIGHBORHOOD_STABLE"),
    ]
    for stats, gate in cases:
        v = gate_verdict(stats)
        assert v["passed"] is False and gate in v["failed"], gate


def test_gate_verdict_nan_inputs_fail_not_pass():
    from ashare_gauntlet.entry_model import gate_verdict

    # Codex P1:切片/回撤缺数据(NaN)不得静默视为满足
    v1 = gate_verdict(_stats(up_diff=float("nan")))
    assert "REGIME_CONSISTENT" in v1["failed"]
    v2 = gate_verdict(_stats(mdd_rule=float("nan")))
    assert "MDD_NOT_WORSE" in v2["failed"]


def test_gate_verdict_missing_key_fails_loud():
    from ashare_gauntlet.entry_model import gate_verdict

    s = _stats()
    del s["coverage"]
    with pytest.raises(KeyError):
        gate_verdict(s)
