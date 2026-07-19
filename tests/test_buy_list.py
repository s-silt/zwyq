"""buy_list 端到端(spec §12 验收):fail-loud、状态唯一、确定性、契约字段。"""
from __future__ import annotations

import json

import pytest

import scripts.buy_list as bl


def _setup(tmp_path, monkeypatch, snap_date="20260101", rows=None, holdings=None):
    cache = tmp_path / "cache"
    for ep in ("daily", "adj_factor", "daily_basic"):
        (cache / ep).mkdir(parents=True, exist_ok=True)
        (cache / ep / f"{snap_date}.parquet").write_bytes(b"")
    fdir = tmp_path / "holdscore"
    fdir.mkdir(exist_ok=True)
    rows = rows if rows is not None else []
    (fdir / f"{snap_date}_factor.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    hpath = tmp_path / "holdings.json"
    holdings = holdings if holdings is not None else {"positions": [], "cash": 100000.0}
    hpath.write_text(json.dumps(holdings, ensure_ascii=False), encoding="utf-8")
    ppath = tmp_path / "policy.json"
    ppath.write_text(json.dumps({"policy_version": "1", "target_positions": 10,
                                 "target_weight": 0.10, "industry_cap": 0.20,
                                 "lot_size": 100, "min_cash": 0}), encoding="utf-8")
    opath = tmp_path / "overrides.json"
    opath.write_text(json.dumps({"overrides": [
        {"ts_code": "600001.SH", "as_of": "20251201", "verdict": "clear",
         "reason": "t", "expires_on": "20270101"}]}, ensure_ascii=False), encoding="utf-8")
    ddir = tmp_path / "decisions"
    monkeypatch.setattr(bl, "CACHE", str(cache))
    monkeypatch.setattr(bl, "FACTOR_DIR", str(fdir))
    monkeypatch.setattr(bl, "HOLDINGS_PATH", str(hpath))
    monkeypatch.setattr(bl, "POLICY_PATH", str(ppath))
    monkeypatch.setattr(bl, "OVERRIDES_PATH", str(opath))
    monkeypatch.setattr(bl, "DECISION_DIR", str(ddir))
    return ddir / f"{snap_date}_buy_decisions.json"


def _row(ts="600001.SH", **kw):
    base = {"ts_code": ts, "name": "甲", "industry": "化工原料", "decile": 10,
            "tier": "🟢", "spec_crowd": False, "spike_limit": False,
            "score": 0.9, "last": 10.0, "f_EP": 0.9, "f_BP": 0.9, "f_IVOL": 0.9}
    base.update(kw)
    return base


def test_stale_snapshot_fails_loud(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, snap_date="20260101")
    with pytest.raises(SystemExit, match="陈旧|无 factor"):
        bl.main(["--as-of", "20251230"])


def test_duplicate_holdings_fail_loud(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "positions": [
            {"ts_code": "600009.SH", "name": "乙", "industry": "航空", "shares": 100,
             "cost": 5.0, "last": 6.0, "mv": 600.0, "stop": 4.0},
            {"ts_code": "600009.SH", "name": "乙", "industry": "航空", "shares": 100,
             "cost": 5.0, "last": 6.0, "mv": 600.0, "stop": 4.0}],
        "cash": 1000.0})
    with pytest.raises(SystemExit, match="重复"):
        bl.main([])


def test_end_to_end_states_and_determinism(tmp_path, monkeypatch, capsys):
    out_path = _setup(
        tmp_path, monkeypatch,
        rows=[_row(),                                     # 完美候选(有覆盖)→ BUY
              _row(ts="600002.SH", name="乙", tier="🟡"),  # 🟡 → WAIT
              _row(ts="600003.SH", name="丙")],            # 无覆盖 → WAIT FACTCHECK_REQUIRED
        holdings={"positions": [
            {"ts_code": "600009.SH", "name": "破线股", "industry": "航空", "shares": 100,
             "cost": 5.0, "last": 3.9, "mv": 390.0, "stop": 4.0}],   # last≤stop → EXIT
            "cash": 100000.0})
    bl.main([])
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["entry_model_version"] == "research-only"
    st = {d["ts_code"]: d for d in out["decisions"]}
    assert st["600001.SH"]["state"] == "BUY" and st["600001.SH"]["execution"]["shares"] > 0
    assert st["600001.SH"]["execution"]["max_entry_price"] is None   # 无准入价格规则不伪造买点
    assert st["600002.SH"]["state"] == "WAIT" and "TIER_NOT_GREEN" in st["600002.SH"]["reason_codes"]
    assert st["600003.SH"]["state"] == "WAIT" and "FACTCHECK_REQUIRED" in st["600003.SH"]["reason_codes"]
    assert st["600009.SH"]["state"] == "EXIT" and "RISK_LINE_BREACH" in st["600009.SH"]["reason_codes"]
    # 每只状态唯一
    assert len(out["decisions"]) == len(st)
    # 确定性:重跑 decisions 逐字节一致(generated_at 除外)
    first = json.dumps(out["decisions"], ensure_ascii=False, sort_keys=True)
    bl.main([])
    out2 = json.loads(out_path.read_text(encoding="utf-8"))
    assert json.dumps(out2["decisions"], ensure_ascii=False, sort_keys=True) == first


def test_snapshot_duplicate_ts_code_fails_loud(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rows=[_row(), _row()])
    with pytest.raises(SystemExit, match="重复"):
        bl.main([])


def test_nonfinite_score_or_bad_shares_fail_loud(tmp_path, monkeypatch):
    import math
    _setup(tmp_path, monkeypatch, rows=[_row(score=math.nan)])
    with pytest.raises(SystemExit, match="非法|无效"):
        bl.main([])
    _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "positions": [{"ts_code": "600009.SH", "name": "乙", "industry": "航空",
                       "shares": 100.5, "cost": 5.0, "last": 6.0, "mv": 600.0, "stop": 4.0}],
        "cash": 1000.0})
    with pytest.raises(SystemExit, match="非法"):
        bl.main([])


def test_held_missing_from_snapshot_with_red_override_exits(tmp_path, monkeypatch):
    # 对抗审查 P0:持仓股掉出 snapshot(恰是最危险情形)时红灯覆盖仍须触发 EXIT
    out_path = _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "positions": [{"ts_code": "600666.SH", "name": "掉档股", "industry": "医药",
                       "shares": 100, "cost": 8.0, "last": 7.0, "mv": 700.0, "stop": 5.0}],
        "cash": 10000.0})
    opath = tmp_path / "overrides.json"
    opath.write_text(json.dumps({"overrides": [
        {"ts_code": "600666.SH", "as_of": "20251201", "verdict": "red",
         "reason": "立案", "expires_on": "20270101"}]}, ensure_ascii=False), encoding="utf-8")
    bl.main([])
    out = json.loads(out_path.read_text(encoding="utf-8"))
    d = {x["ts_code"]: x for x in out["decisions"]}["600666.SH"]
    assert d["state"] == "EXIT" and "GOVERNANCE_RED" in d["reason_codes"]


def test_invalid_override_verdict_fails_loud(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rows=[_row()])
    opath = tmp_path / "overrides.json"
    opath.write_text(json.dumps({"overrides": [
        {"ts_code": "600001.SH", "as_of": "20251201", "verdict": "Red",
         "reason": "笔误大小写", "expires_on": "20270101"}]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="白名单"):
        bl.main([])


def test_holdings_missing_industry_fails_loud(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "positions": [{"ts_code": "600009.SH", "name": "乙", "shares": 100,
                       "cost": 5.0, "last": 6.0, "mv": 600.0, "stop": 4.0}],
        "cash": 1000.0})
    with pytest.raises(SystemExit, match="industry"):
        bl.main([])


def test_empty_buy_is_honest(tmp_path, monkeypatch, capsys):
    out_path = _setup(tmp_path, monkeypatch,
                      rows=[_row(ts="600002.SH", name="乙", spec_crowd=True)])
    bl.main([])
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert all(d["state"] != "BUY" for d in out["decisions"])   # 空 BUY 合法,不降门槛
