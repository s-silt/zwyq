"""buy_list 端到端(spec §12 验收):fail-loud、状态唯一、确定性、契约字段。"""
from __future__ import annotations

import json

import pytest

import scripts.buy_list as bl
from ashare_gauntlet.account_state import AccountFreshnessError
from ashare_gauntlet.c2_review import advance_review, initial_state, record_blocked_review


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
    holdings = holdings if holdings is not None else {"as_of": snap_date, "positions": [], "cash": 100000.0}
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
    ddir.mkdir(exist_ok=True)
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
            "score": 0.9, "last": 10.0, "mv": 400_000.0,   # X-08:mv 为必备字段(万元)
            "f_EP": 0.9, "f_BP": 0.9, "f_IVOL": 0.9}
    base.update(kw)
    return base


def _c2_state(*, blocked=False):
    def evidence(period, as_of, observations):
        return {
            "period": period,
            "as_of": as_of,
            "decision_snapshot": {"path": f"{as_of}_buy_decisions.json", "sha256": "d" * 64},
            "factor_snapshot": {"path": f"{as_of}_factor.json", "sha256": "f" * 64},
            "observations": observations,
        }

    state, _ = advance_review(initial_state(), evidence(
        "202601", "20260130", [{"ts_code": "600001.SH", "name": "甲", "status": "OUTSIDE"}]))
    state, _ = advance_review(state, evidence(
        "202602", "20260227", [{"ts_code": "600001.SH", "name": "甲", "status": "OUTSIDE"}]))
    state, _ = advance_review(state, evidence(
        "202603", "20260331", [
            {"ts_code": "600001.SH", "name": "甲", "status": "OUTSIDE"},
            {"ts_code": "600002.SH", "name": "乙", "status": "OUTSIDE"},
        ]))
    if blocked:
        state = record_blocked_review(state, period="202604", as_of="20260430",
                                      issues=["CORE_EOD_MISSING", "ADJ_FACTOR_GAP"],
                                      evidence_hashes={})
    return state


def test_load_c2_projection_missing_is_normal(tmp_path):
    assert bl.load_c2_projection(tmp_path / "c2_review_state.json") == {
        "status": "NOT_INITIALIZED",
        "last_valid_review_as_of": None,
        "watch": [],
        "exit_eligible": [],
        "error": None,
    }


def test_load_c2_projection_projects_sorted_valid_positions(tmp_path):
    path = tmp_path / "c2_review_state.json"
    path.write_text(json.dumps(_c2_state()), encoding="utf-8")
    projection = bl.load_c2_projection(path)
    assert projection["status"] == "AVAILABLE"
    assert projection["last_valid_review_as_of"] == "20260331"
    assert projection["watch"] == ["600002.SH"]
    assert projection["exit_eligible"] == ["600001.SH"]
    assert all(isinstance(code, str) for code in projection["watch"] + projection["exit_eligible"])


def test_load_c2_projection_blocked_retains_prior_positions_and_issues(tmp_path):
    path = tmp_path / "c2_review_state.json"
    path.write_text(json.dumps(_c2_state(blocked=True)), encoding="utf-8")
    projection = bl.load_c2_projection(path)
    assert projection["status"] == "REVIEW_BLOCKED_DATA"
    assert projection["last_valid_review_as_of"] == "20260331"
    assert projection["watch"] == ["600002.SH"]
    assert projection["exit_eligible"] == ["600001.SH"]
    assert projection["error"] == "REVIEW_BLOCKED_DATA:ADJ_FACTOR_GAP,CORE_EOD_MISSING"


def test_load_c2_projection_rejects_nonfinite_json(tmp_path):
    path = tmp_path / "c2_review_state.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    projection = bl.load_c2_projection(path)
    assert projection["status"] == "UNAVAILABLE"
    assert projection["exit_eligible"] == []
    assert projection["error"] == "C2_STATE_INVALID_JSON"


def test_load_c2_projection_unreadable_oserror(tmp_path):
    path = tmp_path / "c2_review_state.json"
    path.mkdir()

    projection = bl.load_c2_projection(path)

    assert projection["status"] == "UNAVAILABLE"
    assert projection["exit_eligible"] == []
    assert projection["error"] == "C2_STATE_UNREADABLE"


def test_load_c2_projection_rejects_invalid_schema(tmp_path):
    path = tmp_path / "c2_review_state.json"
    path.write_text("{}", encoding="utf-8")

    projection = bl.load_c2_projection(path)

    assert projection["status"] == "UNAVAILABLE"
    assert projection["exit_eligible"] == []
    assert projection["error"] == "C2_STATE_INVALID_SCHEMA"


def test_valid_c2_only_confirmed_position_exits(tmp_path, monkeypatch):
    out_path = _setup(tmp_path, monkeypatch, rows=[
        _row("600001.SH"), _row("600002.SH"),
    ], holdings={"as_of": "20260101", "positions": [
        {"ts_code": "600001.SH", "name": "甲", "industry": "化工原料",
         "shares": 100, "cost": 5.0, "last": 6.0, "mv": 600.0},
        {"ts_code": "600002.SH", "name": "乙", "industry": "化工原料",
         "shares": 100, "cost": 5.0, "last": 6.0, "mv": 600.0},
    ], "cash": 100000.0})
    sidecar = out_path.parent / "c2_review_state.json"
    sidecar.write_text(json.dumps(_c2_state()), encoding="utf-8")
    bl.main([])
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["c2_state"]["status"] == "AVAILABLE"
    decisions = {row["ts_code"]: row for row in out["decisions"]}
    assert decisions["600001.SH"]["state"] == "EXIT"
    assert decisions["600001.SH"]["reason_codes"] == ["EXIT_RULE_C2_CONFIRMED"]
    assert decisions["600002.SH"]["state"] == "HOLD"


def test_unheld_c2_eligible_code_does_not_create_decision(tmp_path, monkeypatch):
    out_path = _setup(tmp_path, monkeypatch, rows=[_row("600002.SH")], holdings={
        "as_of": "20260101", "positions": [{
            "ts_code": "600002.SH", "name": "乙", "industry": "化工原料",
            "shares": 100, "cost": 5.0, "last": 6.0, "mv": 600.0,
        }], "cash": 100000.0,
    })
    sidecar = out_path.parent / "c2_review_state.json"
    sidecar.write_text(json.dumps(_c2_state()), encoding="utf-8")

    bl.main([])

    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert {row["ts_code"] for row in out["decisions"]} == {"600002.SH"}
    assert out["decisions"][0]["state"] == "HOLD"


def test_corrupt_c2_writes_snapshot_then_exits_one(tmp_path, monkeypatch):
    out_path = _setup(tmp_path, monkeypatch, rows=[_row("600009.SH", last=3.0)], holdings={
        "as_of": "20260101", "positions": [{
            "ts_code": "600009.SH", "name": "破线股", "industry": "航空",
            "shares": 100, "cost": 5.0, "last": 3.0, "mv": 300.0, "stop": 4.0,
        }], "cash": 100000.0,
    })
    sidecar = out_path.parent / "c2_review_state.json"
    sidecar.write_bytes(b"{not-json")
    with pytest.raises(SystemExit) as exc:
        bl.main([])
    assert exc.value.code == 1
    assert sidecar.read_bytes() == b"{not-json"
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["data_status"] == "degraded"
    assert out["c2_state"] == {
        "status": "UNAVAILABLE", "last_valid_review_as_of": None,
        "watch": [], "exit_eligible": [], "error": "C2_STATE_INVALID_JSON",
    }
    decision = out["decisions"][0]
    assert decision["state"] == "EXIT"
    assert "RISK_LINE_BREACH" in decision["reason_codes"]


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
        holdings={"as_of": "20260101", "positions": [
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
        "as_of": "20260101",
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


def test_load_overrides_rejects_duplicate_or_malformed_rows(tmp_path):
    path = tmp_path / "overrides.json"
    row = {"ts_code": "600001.SH", "as_of": "20260101",
           "verdict": "clear", "expires_on": "20270101"}
    path.write_text(json.dumps({"overrides": [row, dict(row)]}), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        bl.load_overrides(str(path))

    path.write_text(json.dumps({"overrides": [{"ts_code": "600001.SH"}]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="缺字段"):
        bl.load_overrides(str(path))


def test_load_overrides_rejects_unpadded_and_impossible_dates(tmp_path):
    path = tmp_path / "overrides.json"
    row = {"ts_code": "600001.SH", "as_of": "20260101",
           "verdict": "clear", "expires_on": "20270101"}
    path.write_text(json.dumps({"overrides": [dict(row, expires_on="202695")]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="YYYYMMDD"):
        bl.load_overrides(str(path))

    path.write_text(json.dumps({"overrides": [dict(row, as_of="20260230")]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="YYYYMMDD"):
        bl.load_overrides(str(path))

    path.write_text(json.dumps({"overrides": [dict(row, expires_on="20251231")]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="不能早于"):
        bl.load_overrides(str(path))

    path.write_text(json.dumps({"overrides": [dict(row, verdict=True)]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="白名单"):
        bl.load_overrides(str(path))

    path.write_text(json.dumps({"overrides": [row]}), encoding="utf-8")
    loaded = bl.load_overrides(str(path))
    assert loaded["600001.SH"]["verdict"] == "clear"


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


# ── P0-1: 账户状态门禁 ──

def test_aligned_account_passes_and_outputs_account_fields(tmp_path, monkeypatch):
    """as_of aligned → 门禁通过,输出含 account_as_of / account_source_schema。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "as_of": "20260101",
        "positions": [],
        "cash": 100000.0,
    })
    bl.main([])
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["account_as_of"] == "20260101"
    assert out["account_source_schema"] == "legacy_unversioned"


def test_missing_account_as_of_fails_loud_before_write(tmp_path, monkeypatch):
    """holdings 无 as_of → normalize 后 freshness=MISSING → fail-loud,不写 decision。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "positions": [],
        "cash": 100000.0,
    })
    with pytest.raises((SystemExit, AccountFreshnessError), match="ACCOUNT_AS_OF_MISSING"):
        bl.main([])
    # 不应创建 decision 文件
    assert not out_path.exists()


def test_stale_account_as_of_fails_loud_before_write(tmp_path, monkeypatch):
    """holdings as_of 早于 snapshot → STALE → fail-loud。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "as_of": "20251231",
        "positions": [],
        "cash": 100000.0,
    })
    with pytest.raises((SystemExit, AccountFreshnessError), match="ACCOUNT_AS_OF_STALE"):
        bl.main([])
    assert not out_path.exists()


def test_future_account_as_of_fails_loud_before_write(tmp_path, monkeypatch):
    """holdings as_of 晚于 snapshot → FUTURE → fail-loud。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "as_of": "20260102",
        "positions": [],
        "cash": 100000.0,
    })
    with pytest.raises((SystemExit, AccountFreshnessError), match="ACCOUNT_AS_OF_FUTURE"):
        bl.main([])
    assert not out_path.exists()


def test_invalid_date_account_as_of_fails_loud_before_write(tmp_path, monkeypatch):
    """holdings as_of 非法日期 → INVALID → fail-loud。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "as_of": "20260132",
        "positions": [],
        "cash": 100000.0,
    })
    with pytest.raises((SystemExit, AccountFreshnessError), match="ACCOUNT_AS_OF_INVALID"):
        bl.main([])
    assert not out_path.exists()


def test_existing_decision_not_overwritten_on_fail(tmp_path, monkeypatch):
    """门禁失败不覆盖已有旧 decision 文件。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[_row()], holdings={
        "as_of": "20251231",  # stale
        "positions": [],
        "cash": 100000.0,
    })
    # 先手工创建一个旧文件
    out_path.parent.mkdir(parents=True, exist_ok=True)
    old_content = '{"old": true}'
    out_path.write_text(old_content, encoding="utf-8")
    with pytest.raises((SystemExit, AccountFreshnessError)):
        bl.main([])
    assert out_path.read_text(encoding="utf-8") == old_content


def test_invalid_decision_code_preserves_existing_snapshot_bytes(tmp_path, monkeypatch):
    out_path = _setup(tmp_path, monkeypatch, rows=[_row("A")])
    old = b'{"old": true}\r\n'
    out_path.write_bytes(old)

    with pytest.raises(ValueError, match="invalid ts_code"):
        bl.main([])

    assert out_path.read_bytes() == old
    assert not list(out_path.parent.glob(".tmp_buy_decisions_*"))


def test_snapshot_serialization_failure_preserves_existing_bytes(tmp_path, monkeypatch):
    out_path = _setup(tmp_path, monkeypatch)
    old = b'{"old": true}\r\n'
    out_path.write_bytes(old)
    monkeypatch.setattr(bl, "load_c2_projection", lambda path: {
        "status": "AVAILABLE",
        "last_valid_review_as_of": None,
        "watch": [],
        "exit_eligible": [],
        "error": float("nan"),
    })

    with pytest.raises(ValueError, match="Out of range float values"):
        bl.main([])

    assert out_path.read_bytes() == old
    assert not list(out_path.parent.glob(".tmp_buy_decisions_*"))


@pytest.mark.parametrize("failure_point", ["write", "fsync", "replace"])
def test_snapshot_atomic_failure_preserves_existing_bytes(
        tmp_path, monkeypatch, failure_point):
    out_path = _setup(tmp_path, monkeypatch)
    old = b'{"old": true}\r\n'
    out_path.write_bytes(old)

    if failure_point == "write":
        class FailingWriter:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def write(self, payload):
                raise OSError("write failed")

        def fail_write(fd, *args, **kwargs):
            bl.os.close(fd)
            return FailingWriter()

        monkeypatch.setattr(bl.os, "fdopen", fail_write)
    elif failure_point == "fsync":
        monkeypatch.setattr(
            bl.os, "fsync",
            lambda fd: (_ for _ in ()).throw(OSError("fsync failed")),
        )
    else:
        monkeypatch.setattr(
            bl.os, "replace",
            lambda src, dst: (_ for _ in ()).throw(OSError("replace failed")),
        )

    with pytest.raises(OSError, match=f"{failure_point} failed"):
        bl.main([])

    assert out_path.read_bytes() == old
    assert not list(out_path.parent.glob(".tmp_buy_decisions_*"))


# ---------- X-14:B8 带保留成员状态机(生产候选池 = 当期 D10 ∪ 上期成员∩D8+) ----------

def _two_overrides(tmp_path):
    (tmp_path / "overrides.json").write_text(json.dumps({"overrides": [
        {"ts_code": "600001.SH", "as_of": "20251201", "verdict": "clear",
         "reason": "t", "expires_on": "20270101"},
        {"ts_code": "600002.SH", "as_of": "20251201", "verdict": "clear",
         "reason": "t", "expires_on": "20270101"},
    ]}, ensure_ascii=False), encoding="utf-8")


def test_b8_band_member_from_state_becomes_buy(tmp_path, monkeypatch):
    """上期成员当期 D9(带内)→ 消费为 BUY 候选,来源码 B8_BAND,状态原子推进。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[
        _row("600001.SH"), _row("600002.SH", decile=9, score=0.8)])
    _two_overrides(tmp_path)
    (out_path.parent / "b8_state.json").write_text(json.dumps({
        "schema": "b8_state.v1", "last_as_of": "20251231",
        "prev_members": [], "members": ["600002.SH"]}), encoding="utf-8")
    bl.main([])
    out = json.loads(out_path.read_text(encoding="utf-8"))
    d = {r["ts_code"]: r for r in out["decisions"]}
    assert d["600002.SH"]["state"] == "BUY"
    assert d["600002.SH"]["reason_codes"] == ["B8_BAND", "TIER_GREEN", "FACTCHECK_CLEAR"]
    assert d["600002.SH"]["evidence"]["decile"] == 9
    state = json.loads((out_path.parent / "b8_state.json").read_text(encoding="utf-8"))
    assert state == {"schema": "b8_state.v1", "last_as_of": "20260101",
                     "prev_members": ["600002.SH"],
                     "members": ["600001.SH", "600002.SH"]}


def test_b8_same_day_rerun_uses_prev_members_as_base(tmp_path, monkeypatch):
    """同日重跑以 prev_members 为基数(幂等,不叠加推进):run1 新进 D10 的票在
    run2 前跌到 D9 → 不被当作带成员保留(它不在上期基数里)。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[_row("600001.SH")])
    _two_overrides(tmp_path)
    bl.main([])                      # 首跑:members=[600001.SH](D10 新进)
    # 快照更新(同日):600001 跌到 D9
    (tmp_path / "holdscore" / "20260101_factor.json").write_text(
        json.dumps([_row("600001.SH", decile=9)], ensure_ascii=False), encoding="utf-8")
    bl.main([])                      # 重跑:base=prev=[] → 600001 不保留
    state = json.loads((out_path.parent / "b8_state.json").read_text(encoding="utf-8"))
    assert state["members"] == []
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert all(r["ts_code"] != "600001.SH" or r["state"] != "BUY" for r in out["decisions"])


def test_b8_dropped_out_of_band_member_is_not_consumed(tmp_path, monkeypatch):
    """上期成员跌穿带(D7)→ 不进候选池,决策名单不含它(消费面=D10∪带∪持仓)。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[
        _row("600001.SH"), _row("600002.SH", decile=7, score=0.3)])
    _two_overrides(tmp_path)
    (out_path.parent / "b8_state.json").write_text(json.dumps({
        "schema": "b8_state.v1", "last_as_of": "20251231",
        "prev_members": [], "members": ["600002.SH"]}), encoding="utf-8")
    bl.main([])
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert all(r["ts_code"] != "600002.SH" for r in out["decisions"])
    state = json.loads((out_path.parent / "b8_state.json").read_text(encoding="utf-8"))
    assert state["members"] == ["600001.SH"]


def test_b8_state_missing_seeds_from_d10(tmp_path, monkeypatch):
    """无状态文件=未初始化:首跑成员=当期 D10(与旧口径一致),并落盘状态。"""
    out_path = _setup(tmp_path, monkeypatch, rows=[
        _row("600001.SH"), _row("600002.SH", decile=9, score=0.8)])
    _two_overrides(tmp_path)
    bl.main([])
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert {r["ts_code"] for r in out["decisions"]} == {"600001.SH"}
    state = json.loads((out_path.parent / "b8_state.json").read_text(encoding="utf-8"))
    assert state["members"] == ["600001.SH"] and state["prev_members"] == []


def test_b8_as_of_regression_fails_loud(tmp_path, monkeypatch):
    out_path = _setup(tmp_path, monkeypatch, rows=[_row("600001.SH")])
    (out_path.parent / "b8_state.json").write_text(json.dumps({
        "schema": "b8_state.v1", "last_as_of": "20260102",
        "prev_members": [], "members": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="不可倒退"):
        bl.main([])


def test_b8_invalid_state_schema_fails_loud(tmp_path, monkeypatch):
    out_path = _setup(tmp_path, monkeypatch, rows=[_row("600001.SH")])
    (out_path.parent / "b8_state.json").write_text(json.dumps({
        "schema": "other.v9", "members": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema 非法"):
        bl.main([])
