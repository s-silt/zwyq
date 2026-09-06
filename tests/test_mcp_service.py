import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_gauntlet import mcp_service as svc


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_validate_codes_normalizes_and_rejects_injection() -> None:
    assert svc.validate_codes(["001218.sz", "600875.SH"]) == ["001218.SZ", "600875.SH"]
    with pytest.raises(ValueError, match="invalid ts_code"):
        svc.validate_codes(["001218.SZ; whoami"])


def test_read_json_cannot_escape_root(tmp_path: Path) -> None:
    dump(tmp_path / "safe.json", {"ok": True})
    assert svc.read_json("safe.json", tmp_path) == {"ok": True}
    with pytest.raises(ValueError, match="escapes"):
        svc.read_json("../outside.json", tmp_path)


def test_latest_snapshots_ignore_non_date_files(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdscore/20260807_factor.json", [])
    dump(tmp_path / "data/holdscore/latest_factor.json", [{"bad": True}])
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", {"decisions": []})
    assert svc.latest_factor_path(tmp_path).name == "20260807_factor.json"
    assert svc.latest_decision_path(tmp_path).name == "20260807_buy_decisions.json"


def test_account_snapshot_calculates_total_and_industry_weight(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260807", "cash": 1000, "conditional_orders": "2张",
        "positions": [
            {"ts_code": "000001.SZ", "mv": 9000, "industry": "银行", "bucket": "长线"},
            {"ts_code": "000002.SZ", "mv": 10000, "industry": "地产", "bucket": "短线"},
        ],
    })
    result = svc.account_snapshot(tmp_path)
    assert result["total_assets"] == 20000
    assert result["industry_weights"]["地产"]["weight"] == 0.5
    assert result["short_position"]["ts_code"] == "000002.SZ"


def test_strategy_context_is_bounded_and_omits_manual_details(tmp_path: Path) -> None:
    dump(tmp_path / "data/trading_policy.json", {
        "policy_version": "2", "target_positions": 10, "target_weight": 0.1,
        "industry_cap": 0.2, "lot_size": 100, "min_cash": 5000,
        "note": "PRIVATE_POLICY_NOTE",
    })
    dump(tmp_path / "data/profile.json", {
        "as_of": "20260807", "excluded_industries": ["半导体"],
        "note": "PRIVATE_PROFILE_NOTE",
    })
    dump(tmp_path / "data/factcheck_overrides.json", {
        "note": "PRIVATE_FACTCHECK_NOTE",
        "overrides": [
            {"ts_code": "600001.SH", "as_of": "20260801", "verdict": "clear",
             "expires_on": "20261101", "reason": "PRIVATE_FACTCHECK_REASON"},
            {"ts_code": "600002.SH", "as_of": "20260802", "verdict": "red",
             "expires_on": "20261102", "reason": "PRIVATE_RED_REASON"},
        ],
    })
    dump(tmp_path / "data/trigger_bands.json", {
        "note": "PRIVATE_BAND_NOTE",
        "items": [{"ts_code": "600001.SH", "band_low": 1, "band_high": 2}],
        "retired": [{"ts_code": "600002.SH"}],
    })

    result = svc.strategy_context(tmp_path)
    encoded = json.dumps(result, ensure_ascii=False)
    assert result["policy"] == {
        "policy_version": "2", "target_positions": 10, "target_weight": 0.1,
        "industry_cap": 0.2, "lot_size": 100, "min_cash": 5000,
    }
    assert result["profile"]["excluded_industries"] == ["半导体"]
    assert result["factchecks"] == {
        "override_count": 2, "verdict_counts": {"clear": 1, "red": 1},
        "latest_as_of": "20260802", "details_included": False,
    }
    assert result["trigger_bands"] == {
        "active_count": 1, "retired_count": 1, "details_included": False,
    }
    assert "PRIVATE_" not in encoded
    assert "ts_code" not in result["factchecks"]


def test_refresh_eod_rejects_oversized_or_future_range(monkeypatch) -> None:
    with pytest.raises(ValueError, match="31"):
        svc.refresh_eod("20260101", "20260215")
    future = str(int(__import__("datetime").date.today().strftime("%Y%m%d")) + 10000)
    with pytest.raises((ValueError, OverflowError)):
        svc.refresh_eod(future, future)


def test_run_module_rejects_arbitrary_command() -> None:
    with pytest.raises(ValueError, match="allow-listed"):
        svc.run_module("os.system", ["whoami"])


def test_stock_brief_surfaces_quote_failure(tmp_path: Path, monkeypatch) -> None:
    dump(tmp_path / "data/holdscore/20260807_factor.json",
         [{"ts_code": "001218.SZ", "score": 1.0, "decile": 10}])
    dump(tmp_path / "data/holdings.json", {"cash": 1, "positions": []})
    dump(tmp_path / "data/factcheck_overrides.json", {"overrides": []})
    dump(tmp_path / "data/trigger_bands.json", {"items": []})
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", {
        "as_of": "20260807", "data_status": "complete",
        "c2_state": {
            "status": "NOT_INITIALIZED", "last_valid_review_as_of": None,
            "watch": [], "exit_eligible": [], "error": None,
        },
        "decisions": [{
            "ts_code": "001218.SZ", "state": "WAIT", "reason_codes": [],
            "evidence": {}, "execution": {"shares": 0}, "invalidations": [],
        }],
    })
    monkeypatch.setattr(svc, "intraday_quotes", lambda _: (_ for _ in ()).throw(OSError("down")))
    result = svc.stock_brief("001218.SZ", root=tmp_path)
    assert result["factor"]["decile"] == 10
    assert "OSError: down" in result["intraday_error"]
    assert result["machine_decision"]["status"] == "found"
    assert result["actionable_view"]["user_action"] == "WAIT"


def _c2_not_initialized() -> dict:
    return {
        "status": "NOT_INITIALIZED",
        "last_valid_review_as_of": None,
        "watch": [],
        "exit_eligible": [],
        "error": None,
    }


def _decision_snapshot(state: str = "WAIT", max_entry_price=None) -> dict:
    reasons = ["D10", "FACTCHECK_CLEAR"] if state == "BUY" else []
    return {
        "as_of": "20260807",
        "data_status": "complete",
        "c2_state": _c2_not_initialized(),
        "decisions": [{
            "ts_code": "001218.SZ",
            "state": state,
            "reason_codes": reasons,
            "evidence": {"decile": 10},
            "execution": {
                "eligible_from": "NEXT_TRADING_DAY" if state == "BUY" else None,
                "max_entry_price": max_entry_price,
                "shares": 100,
            },
            "invalidations": [],
        }],
    }


def _write_ready_service_fixture(
    root: Path, *, c2_state: dict, data_status: str = "complete",
) -> None:
    for endpoint in ("daily", "adj_factor", "daily_basic", "stk_limit"):
        path = root / "data/cache" / endpoint / "20260807.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    dump(root / "data/holdscore/20260807_factor.json", [])
    snapshot = _decision_snapshot()
    snapshot["data_status"] = data_status
    snapshot["c2_state"] = c2_state
    dump(root / "data/decisions/20260807_buy_decisions.json", snapshot)
    dump(root / "data/holdings.json", {
        "as_of": "20260807", "cash": 1000, "positions": [],
        "conditional_orders": {
            "schema_version": 2,
            "orders": [{
                "order_id": "ord-001", "ts_code": "000001.SZ", "side": "BUY",
                "condition": {"field": "close", "operator": "<="},
                "price": 10.5, "shares": 100, "valid_from": "20260801",
                "valid_until": "20260831", "status": "active",
            }],
        },
    })
    dump(root / "data/trading_policy.json", {})
    dump(root / "data/profile.json", {})
    dump(root / "data/factcheck_overrides.json", {"overrides": []})


def test_latest_decisions_validates_snapshot_contract(tmp_path: Path) -> None:
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", _decision_snapshot())
    assert svc.latest_decisions(tmp_path)["as_of"] == "20260807"

    bad = _decision_snapshot()
    bad["as_of"] = "20260806"
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", bad)
    with pytest.raises(ValueError, match="mismatch"):
        svc.latest_decisions(tmp_path)

    bad_code = _decision_snapshot()
    bad_code["decisions"][0]["ts_code"] = "A"
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", bad_code)
    with pytest.raises(ValueError, match="invalid ts_code"):
        svc.latest_decisions(tmp_path)


def test_latest_decisions_rejects_incomplete_or_duplicate_snapshot(tmp_path: Path) -> None:
    for status in ("partial", "degraded"):
        incomplete = _decision_snapshot()
        incomplete["data_status"] = status
        dump(tmp_path / "data/decisions/20260807_buy_decisions.json", incomplete)
        with pytest.raises(ValueError, match="not complete"):
            svc.latest_decisions(tmp_path)

    duplicate = _decision_snapshot()
    duplicate["decisions"].append(dict(duplicate["decisions"][0]))
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        svc.latest_decisions(tmp_path)


def test_latest_decisions_rejects_unavailable_c2_marked_complete(tmp_path: Path) -> None:
    snapshot = _decision_snapshot()
    snapshot["c2_state"] = {
        "status": "UNAVAILABLE",
        "last_valid_review_as_of": None,
        "watch": [],
        "exit_eligible": [],
        "error": "C2_STATE_UNREADABLE",
    }
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)

    with pytest.raises(ValueError, match="not complete"):
        svc.latest_decisions(tmp_path)


@pytest.mark.parametrize(("case", "c2_state"), [
    ("missing", None),
    ("none", None),
    ("empty", {}),
    ("unknown_status", {
        "status": "UNKNOWN", "last_valid_review_as_of": None,
        "watch": [], "exit_eligible": [], "error": None,
    }),
    ("non_string_status", {
        "status": [], "last_valid_review_as_of": None,
        "watch": [], "exit_eligible": [], "error": None,
    }),
    ("available_missing_exit_eligible", {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260807",
        "watch": [], "error": None,
    }),
    ("invalid_date", {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260230",
        "watch": [], "exit_eligible": [], "error": None,
    }),
    ("duplicate_watch", {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260807",
        "watch": ["001218.SZ", "001218.SZ"], "exit_eligible": [], "error": None,
    }),
    ("empty_code", {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260807",
        "watch": [""], "exit_eligible": [], "error": None,
    }),
    ("unsorted_codes", {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260807",
        "watch": ["600875.SH", "001218.SZ"], "exit_eligible": [], "error": None,
    }),
    ("overlapping_codes", {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260807",
        "watch": ["001218.SZ"], "exit_eligible": ["001218.SZ"], "error": None,
    }),
    ("available_error", {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260807",
        "watch": [], "exit_eligible": [], "error": "unexpected",
    }),
    ("available_members_without_last_valid", {
        "status": "AVAILABLE", "last_valid_review_as_of": None,
        "watch": ["A"], "exit_eligible": [], "error": None,
    }),
    ("not_initialized_with_state", {
        "status": "NOT_INITIALIZED", "last_valid_review_as_of": "20260807",
        "watch": ["001218.SZ"], "exit_eligible": [], "error": None,
    }),
    ("not_initialized_with_error", {
        "status": "NOT_INITIALIZED", "last_valid_review_as_of": None,
        "watch": [], "exit_eligible": [], "error": "unexpected",
    }),
    ("blocked_without_error", {
        "status": "REVIEW_BLOCKED_DATA", "last_valid_review_as_of": "20260807",
        "watch": [], "exit_eligible": [], "error": None,
    }),
    ("blocked_unstable_error", {
        "status": "REVIEW_BLOCKED_DATA", "last_valid_review_as_of": "20260807",
        "watch": [], "exit_eligible": [], "error": "CORE_EOD_MISSING",
    }),
    ("blocked_members_without_last_valid", {
        "status": "REVIEW_BLOCKED_DATA", "last_valid_review_as_of": None,
        "watch": [], "exit_eligible": ["A"],
        "error": "REVIEW_BLOCKED_DATA:CORE_EOD_MISSING",
    }),
    ("extra_field", {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260807",
        "watch": [], "exit_eligible": [], "error": None, "extra": True,
    }),
])
def test_latest_decisions_rejects_invalid_c2_projection(
        tmp_path: Path, case: str, c2_state: object) -> None:
    snapshot = _decision_snapshot()
    if case == "missing":
        snapshot.pop("c2_state")
    else:
        snapshot["c2_state"] = c2_state
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)

    with pytest.raises(ValueError, match="c2_state"):
        svc.latest_decisions(tmp_path)


@pytest.mark.parametrize("c2_state", [
    {
        "status": "AVAILABLE", "last_valid_review_as_of": "20260807",
        "watch": ["001218.SZ"], "exit_eligible": ["600875.SH"], "error": None,
    },
    {
        "status": "REVIEW_BLOCKED_DATA", "last_valid_review_as_of": "20260807",
        "watch": ["001218.SZ"], "exit_eligible": ["600875.SH"],
        "error": "REVIEW_BLOCKED_DATA:CORE_EOD_MISSING",
    },
])
def test_latest_decisions_accepts_consumable_c2_projection(
        tmp_path: Path, c2_state: dict) -> None:
    snapshot = _decision_snapshot()
    snapshot["c2_state"] = c2_state
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)

    assert svc.latest_decisions(tmp_path)["as_of"] == "20260807"


def test_latest_decisions_projects_c2_state_and_status(tmp_path: Path) -> None:
    c2_state = {
        "status": "REVIEW_BLOCKED_DATA", "last_valid_review_as_of": "20260731",
        "watch": ["001218.SZ"], "exit_eligible": [],
        "error": "REVIEW_BLOCKED_DATA:CORE_EOD_MISSING",
    }
    snapshot = _decision_snapshot()
    snapshot["c2_state"] = c2_state
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)

    result = svc.latest_decisions(tmp_path, summary_only=True)

    assert result["c2_status"] == "REVIEW_BLOCKED_DATA"
    assert result["c2_state"] == c2_state


def test_latest_decisions_accepts_opaque_code_from_real_c2_state(tmp_path: Path) -> None:
    from ashare_gauntlet.c2_review import advance_review, initial_state
    from ashare_gauntlet.decision_snapshot import validate_c2_projection
    from scripts.buy_list import load_c2_projection

    evidence = {
        "period": "202601",
        "as_of": "20260130",
        "decision_snapshot": {"path": "decision.json", "sha256": "d" * 64},
        "factor_snapshot": {"path": "factor.json", "sha256": "f" * 64},
        "observations": [{"ts_code": "A", "name": "甲", "status": "OUTSIDE"}],
    }
    state, _ = advance_review(initial_state(), evidence)
    sidecar = tmp_path / "c2_review_state.json"
    dump(sidecar, state)
    projection = load_c2_projection(sidecar)
    assert projection["watch"] == ["A"]
    assert validate_c2_projection(projection) is projection

    snapshot = _decision_snapshot()
    snapshot["c2_state"] = projection
    snapshot["decisions"] = []
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)

    assert svc.latest_decisions(tmp_path)["as_of"] == "20260807"


def test_latest_decisions_rejects_buy_without_hard_evidence(tmp_path: Path) -> None:
    snapshot = _decision_snapshot("BUY", 8.2)
    snapshot["decisions"][0]["reason_codes"] = []
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)
    with pytest.raises(ValueError, match="D10/B8_BAND/FACTCHECK_CLEAR"):
        svc.latest_decisions(tmp_path)


def test_latest_decisions_accepts_b8_band_buy(tmp_path: Path) -> None:
    """X-14:B8 带保留成员(decile 8/9 + B8_BAND 码)的 BUY 通过快照契约;
    decile 与来源码不一致仍拒。"""
    snapshot = _decision_snapshot("BUY", 8.2)
    snapshot["decisions"][0]["reason_codes"] = ["B8_BAND", "FACTCHECK_CLEAR"]
    snapshot["decisions"][0]["evidence"]["decile"] = 9
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)
    svc.latest_decisions(tmp_path)   # 不抛即通过

    mismatch = _decision_snapshot("BUY", 8.2)          # decile=10 却带 B8_BAND → 拒
    mismatch["decisions"][0]["reason_codes"] = ["B8_BAND", "FACTCHECK_CLEAR"]
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", mismatch)
    with pytest.raises(ValueError, match="D10/B8_BAND"):
        svc.latest_decisions(tmp_path)


def test_actionable_view_never_promotes_wait_or_invents_prices() -> None:
    wait = svc._actionable_view({"decision": _decision_snapshot()["decisions"][0]})
    assert wait["machine_state"] == "WAIT"
    assert wait["user_action"] == "WAIT"
    assert wait["limit_price"] is None
    assert wait["stop_price"] is None

    buy_without_price = _decision_snapshot("BUY")["decisions"][0]
    view = svc._actionable_view({"decision": buy_without_price})
    assert view["machine_state"] == "BUY"
    assert view["user_action"] == "WAIT"
    assert view["reason"] == "VERIFIED_ENTRY_PRICE_MISSING"
    assert view["buy_range"] is None
    assert view["limit_price"] is None
    assert view["stop_price"] is None

    buy_without_shares = _decision_snapshot("BUY", 8.20)["decisions"][0]
    buy_without_shares["execution"]["shares"] = 0
    view = svc._actionable_view({"decision": buy_without_shares})
    assert view["user_action"] == "WAIT"
    assert view["actionable"] is False
    assert view["reason"] == "EXECUTABLE_SHARES_MISSING"
    assert view["limit_price"] is None


def test_stock_brief_marks_trigger_band_advisory_and_alignment(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdscore/20260806_factor.json",
         [{"ts_code": "001218.SZ", "score": 1.0, "decile": 10}])
    dump(tmp_path / "data/holdings.json", {"cash": 1, "positions": []})
    dump(tmp_path / "data/factcheck_overrides.json", {"overrides": []})
    dump(tmp_path / "data/trigger_bands.json", {
        "items": [{"ts_code": "001218.SZ", "band_low": 8.0, "band_high": 8.2}],
    })
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", _decision_snapshot())

    result = svc.stock_brief("001218.SZ", include_intraday=False, root=tmp_path)
    assert result["trigger_band_status"] == "advisory"
    assert result["trigger_band_can_change_decision"] is False
    assert result["snapshot_alignment"]["factor_matches_decision"] is False
    assert result["actionable_view"]["user_action"] == "WAIT"


def test_stock_brief_explicitly_marks_missing_machine_decision(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdscore/20260807_factor.json",
         [{"ts_code": "001218.SZ", "score": 1.0, "decile": 10}])
    dump(tmp_path / "data/holdings.json", {"cash": 1, "positions": []})
    dump(tmp_path / "data/factcheck_overrides.json", {"overrides": []})
    dump(tmp_path / "data/trigger_bands.json", {"items": []})
    snapshot = _decision_snapshot()
    snapshot["decisions"] = []
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)

    result = svc.stock_brief("001218.SZ", include_intraday=False, root=tmp_path)
    assert result["machine_decision"]["status"] == "missing"
    assert result["actionable_view"]["user_action"] == "WAIT"
    assert result["actionable_view"]["reason"] == "MACHINE_DECISION_MISSING"


def test_refresh_eod_requires_valid_complete_coverage(monkeypatch) -> None:
    complete = {
        "ok": True,
        "calendar_status": "complete",
        "required_endpoints": ["daily", "adj_factor", "daily_basic", "stk_limit"],
        "expected_pairs": 4,
        "completed_pairs": 4,
        "failed_pairs": [],
        "fatal_error": None,
    }
    stdout = svc._BACKFILL_REPORT_PREFIX + json.dumps(complete)
    monkeypatch.setattr(
        svc, "run_module",
        lambda module, args, timeout, root: {
            "ok": True, "module": module, "returncode": 0, "stdout": stdout, "stderr": "",
        },
    )
    result = svc.refresh_eod("20260807", "20260807", root=Path("."))
    assert result["ok"] is True
    assert result["coverage"] == complete

    captured: dict = {}

    def capture_run(module, args, timeout, root):
        captured.update(module=module, args=list(args), timeout=timeout, root=root)
        return {"ok": True, "module": module, "returncode": 0,
                "stdout": stdout, "stderr": ""}

    monkeypatch.setattr(svc, "run_module", capture_run)
    svc.refresh_eod("20260807", "20260807", root=Path("."))
    assert captured["module"] == "scripts.backfill"
    assert captured["args"] == [
        "20260807", "20260807", "--strict-market", "--strict-env", "--report-json",
    ]

    monkeypatch.setattr(
        svc, "run_module",
        lambda module, args, timeout, root: {
            "ok": True, "module": module, "returncode": 0, "stdout": "", "stderr": "",
        },
    )
    result = svc.refresh_eod("20260807", "20260807", root=Path("."))
    assert result["ok"] is False
    assert "coverage report" in result["contract_error"]


def test_account_snapshot_preserves_unknowns_and_detects_short_violation(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260807", "conditional_orders": "待核对",
        "positions": [
            {"ts_code": "000001.SZ", "mv": 0, "industry": "银行", "bucket": "短线"},
            {"ts_code": "000002.SZ", "industry": "地产", "bucket": "短线"},
        ],
    })
    result = svc.account_snapshot(tmp_path)
    assert result["data_status"] == "incomplete"
    assert result["cash"] is None
    assert result["market_value"] is None
    assert result["total_assets"] is None
    assert result["industry_weights"] is None
    assert result["short_slot"]["count"] == 2
    assert result["short_slot"]["violation"] is True
    assert result["conditional_orders"]["status"] == "unverified"


def test_decisions_are_filtered_paginated_and_compact(tmp_path: Path) -> None:
    snapshot = _decision_snapshot()
    snapshot["decisions"] = [
        {"ts_code": f"{index:06d}.SZ", "name": f"N{index}",
         "state": "BUY" if index % 2 else "WAIT",
         "reason_codes": (["D10", "FACTCHECK_CLEAR"] if index % 2 else ["R"]),
         "evidence": {"score": index, "industry": "化工", "decile": 10,
                      "extra": "x" * 1000},
         "execution": {"eligible_from": "NEXT_TRADING_DAY" if index % 2 else None,
                       "max_entry_price": None, "shares": 0},
         "invalidations": ["x"]}
        for index in range(1, 7)
    ]
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)

    summary = svc.latest_decisions(tmp_path, summary_only=True)
    assert summary["decisions"] == []
    assert summary["summary"]["state_counts"]["BUY"] == 3
    page = svc.latest_decisions(tmp_path, states=["BUY"], offset=1, limit=1)
    assert page["page"]["filtered_total"] == 3
    assert page["page"]["has_more"] is True
    assert page["page"]["next_offset"] == 2
    assert "extra" not in page["decisions"][0]["evidence"]
    assert len(json.dumps(svc.latest_decisions(tmp_path), ensure_ascii=False).encode()) < 32768


def test_factor_rows_paginate_and_stock_brief_finds_after_500(tmp_path: Path) -> None:
    rows = [{"ts_code": f"{index:06d}.SZ", "score": 1000 - index,
             "decile": 10, "name": f"N{index}", "extra": "x" * 500}
            for index in range(1, 503)]
    dump(tmp_path / "data/holdscore/20260807_factor.json", rows)
    dump(tmp_path / "data/holdings.json", {"as_of": "20260807", "cash": 0, "positions": []})
    dump(tmp_path / "data/factcheck_overrides.json", {"overrides": []})
    dump(tmp_path / "data/trigger_bands.json", {"items": []})
    snapshot = _decision_snapshot()
    snapshot["decisions"] = []
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", snapshot)

    page = svc.latest_factor_rows(deciles=[10], limit=20, root=tmp_path)
    assert page["page"]["returned"] == 20
    assert page["page"]["has_more"] is True
    assert "extra" not in page["rows"][0]
    assert len(json.dumps(page, ensure_ascii=False).encode()) < 24576
    brief = svc.stock_brief("000502.SZ", include_intraday=False, root=tmp_path)
    assert brief["factor"]["ts_code"] == "000502.SZ"


def test_intraday_quotes_marks_out_of_session_snapshot(monkeypatch) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(svc, "fetch_quotes", lambda codes: {
        codes[0]: {"trade_date": "20260807", "timestamp_status": "valid"}
    })
    result = svc.intraday_quotes(
        ["001218.SZ"], now=datetime(2026, 8, 10, 0, 45, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    assert result["session"] == "pre_open"
    assert result["quote_mode"] == "last_available_snapshot"
    assert result["is_intraday"] is False


def test_intraday_quotes_partial_response_is_not_marked_live(monkeypatch) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(svc, "fetch_quotes", lambda codes: {
        codes[0]: {"trade_date": "20260811", "timestamp_status": "valid"}
    })
    result = svc.intraday_quotes(
        ["001218.SZ", "600875.SH"],
        now=datetime(2026, 8, 11, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    assert result["missing"] == ["600875.SH"]
    assert result["quote_mode"] == "last_available_snapshot"
    assert result["is_intraday"] is False


def test_stock_brief_suspends_band_for_held_symbol_and_ignores_newer_factcheck(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdscore/20260807_factor.json",
         [{"ts_code": "001218.SZ", "score": 1.0, "decile": 10}])
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260807", "cash": 0,
        "positions": [{"ts_code": "001218.SZ", "mv": 1, "bucket": "长线"}],
    })
    dump(tmp_path / "data/factcheck_overrides.json", {"overrides": [
        {"ts_code": "001218.SZ", "as_of": "20260809", "verdict": "clear"},
    ]})
    dump(tmp_path / "data/trigger_bands.json", {
        "items": [{"ts_code": "001218.SZ", "band_low": 8, "band_high": 9}],
    })
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", _decision_snapshot())
    result = svc.stock_brief("001218.SZ", include_intraday=False, root=tmp_path)
    assert result["factcheck"] is None
    assert result["factcheck_alignment"]["status"] == "newer_ignored"
    assert result["trigger_band_status"] == "suspended"
    assert result["trigger_band_reason"] == "HELD_POSITION_SAME_SYMBOL"


def test_healthcheck_separates_operational_from_recommendation_readiness(tmp_path: Path) -> None:
    for endpoint in ("daily", "adj_factor", "daily_basic", "stk_limit"):
        path = tmp_path / "data/cache" / endpoint / "20260807.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    dump(tmp_path / "data/holdscore/20260807_factor.json", [])
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", {
        "as_of": "20260807", "data_status": "complete",
        "c2_state": _c2_not_initialized(), "decisions": [],
    })
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260805", "cash": 0, "conditional_orders": "待核对", "positions": [],
    })
    dump(tmp_path / "data/trading_policy.json", {})
    dump(tmp_path / "data/profile.json", {})
    dump(tmp_path / "data/factcheck_overrides.json", {"overrides": []})
    result = svc.healthcheck(tmp_path)
    assert result["ok"] is True
    readiness = result["recommendation_readiness"]
    assert readiness["ready"] is False
    assert "ACCOUNT_AS_OF_STALE" in readiness["blockers"]
    assert "CONDITIONAL_ORDERS_UNVERIFIED" not in readiness["blockers"]
    assert "CONDITIONAL_ORDERS_INVALID" not in readiness["blockers"]


def test_healthcheck_blocks_review_blocked_c2_snapshot(tmp_path: Path) -> None:
    c2_state = {
        "status": "REVIEW_BLOCKED_DATA", "last_valid_review_as_of": "20260731",
        "watch": ["001218.SZ"], "exit_eligible": [],
        "error": "REVIEW_BLOCKED_DATA:CORE_EOD_MISSING",
    }
    _write_ready_service_fixture(tmp_path, c2_state=c2_state)

    readiness = svc.healthcheck(tmp_path)["recommendation_readiness"]

    assert readiness["ready"] is False
    assert "C2_REVIEW_BLOCKED_DATA" in readiness["blockers"]
    assert readiness["components"]["decision"]["c2_status"] == "REVIEW_BLOCKED_DATA"


def test_readiness_names_unavailable_and_degraded_decision_blockers(
    tmp_path: Path,
) -> None:
    c2_state = {
        "status": "UNAVAILABLE", "last_valid_review_as_of": None,
        "watch": [], "exit_eligible": [], "error": "C2_STATE_UNREADABLE",
    }
    _write_ready_service_fixture(
        tmp_path, c2_state=c2_state, data_status="degraded",
    )

    readiness = svc._recommendation_readiness(tmp_path)

    assert readiness["ready"] is False
    assert "C2_REVIEW_UNAVAILABLE" in readiness["blockers"]
    assert "DECISION_SNAPSHOT_DEGRADED" in readiness["blockers"]


def test_maintenance_stages_run_exactly_one_allowlisted_module(
    tmp_path: Path, monkeypatch,
) -> None:
    calls: list[tuple[str, list[str], int, Path]] = []

    def fake_run(module, args, timeout, root):
        calls.append((module, list(args), timeout, root))
        return {"ok": True, "module": module, "returncode": 0,
                "stdout": "", "stderr": ""}

    monkeypatch.setattr(svc, "run_module", fake_run)
    monkeypatch.setattr(svc, "latest_decisions", lambda *args, **kwargs: {"summary": {}})
    monkeypatch.setattr(svc, "_recommendation_readiness", lambda root: {"ready": False})

    assert svc.generate_factor_snapshot(top=7, root=tmp_path)["stage"] == "factor_rank"
    assert calls == [("scripts.factor_rank", ["--board", "main", "--top", "7"],
                      1200, tmp_path.resolve())]

    calls.clear()
    assert svc.generate_machine_decisions(root=tmp_path)["stage"] == "buy_list"
    assert calls == [("scripts.buy_list", [], 300, tmp_path.resolve())]

    calls.clear()
    assert svc.generate_personal_aggressive_view(
        top=9, root=tmp_path)["stage"] == "aggressive_pick"
    assert calls == [("scripts.aggressive_pick", ["--top", "9"], 300,
                      tmp_path.resolve())]

    calls.clear()
    assert svc.calculate_market_temperature(root=tmp_path)["stage"] == "market_temp"
    assert calls == [("scripts.market_temp", [], 300, tmp_path.resolve())]


@pytest.mark.parametrize("value", [0, 101, True, 1.5])
def test_top_bounded_before_subprocess(value, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(svc, "run_module", lambda *args, **kwargs: pytest.fail("subprocess used"))
    with pytest.raises(ValueError, match="integer in 1..100"):
        svc.generate_factor_snapshot(top=value, root=tmp_path)
    with pytest.raises(ValueError, match="integer in 1..100"):
        svc.generate_personal_aggressive_view(top=value, root=tmp_path)


def test_generate_daily_analysis_reports_optional_failure_as_partial(tmp_path: Path, monkeypatch) -> None:
    calls = iter([True, True, False, True])
    monkeypatch.setattr(svc, "run_module", lambda module, args, timeout, root: {
        "ok": next(calls), "module": module, "returncode": 0, "stdout": "", "stderr": "boom",
    })
    monkeypatch.setattr(svc, "latest_decisions", lambda *args, **kwargs: {"summary": {}})
    monkeypatch.setattr(svc, "account_snapshot", lambda root: {
        "data_status": "complete", "as_of": "20260807", "total_assets": 1,
        "short_slot": {}, "conditional_orders": {"status": "verified"},
    })
    monkeypatch.setattr(svc, "_recommendation_readiness", lambda root: {"ready": True})
    result = svc.generate_daily_analysis(root=tmp_path)
    assert result["ok"] is False
    assert result["status"] == "partial"
    assert result["failed_stages"] == ["aggressive_pick"]
    assert "decisions" not in result


def _governance_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    pledge = pd.DataFrame([{
        "ann_date": "20260731", "holder_name": "股东甲",
        "pledged_amount": 40.0, "holding_amount": 100.0, "h_total_ratio": 10.0,
    }])
    audit = pd.DataFrame([{
        "end_date": "20251231", "ann_date": "20260401",
        "audit_result": "标准无保留意见",
    }])
    return pledge, audit


def _write_governance_cache(root: Path, code: str = "600875.SH",
                            *, pledge: pd.DataFrame | None = None,
                            audit: pd.DataFrame | None = None) -> None:
    default_pledge, default_audit = _governance_frames()
    if pledge is not None:
        default_pledge = pledge
    if audit is not None:
        default_audit = audit
    for endpoint, frame in (("pledge_detail", default_pledge),
                            ("fina_audit", default_audit)):
        path = root / "data/cache" / endpoint / f"{code}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def test_refresh_financials_includes_governance_tables(monkeypatch, tmp_path: Path) -> None:
    from ashare_gauntlet import config
    from ashare_gauntlet.data import fetch

    calls: list[tuple[str, str, Path, bool]] = []
    monkeypatch.setattr(config, "tushare_pro", lambda **kwargs: object())

    def fake_fetch(pro, endpoint, code, cache_dir, force=False):
        calls.append((endpoint, code, Path(cache_dir), force))
        date_column = "ann_date" if endpoint == "pledge_detail" else "end_date"
        return pd.DataFrame([{date_column: "20260731"}])

    monkeypatch.setattr(fetch, "fetch_symbol_table", fake_fetch)
    result = svc.refresh_financials(["600875.SH"], root=tmp_path)
    endpoints = [call[0] for call in calls]
    assert len(calls) == 12
    assert "pledge_detail" in endpoints
    assert "fina_audit" in endpoints
    assert all(call[2] == tmp_path / "data/cache" and call[3] is True for call in calls)
    assert result["codes"]["600875.SH"]["pledge_detail"]["latest"] == "20260731"
    assert result["codes"]["600875.SH"]["fina_audit"]["latest"] == "20260731"
    assert result["ok"] is True


def test_refresh_financials_preserves_other_results_on_failure(monkeypatch, tmp_path: Path) -> None:
    from ashare_gauntlet import config
    from ashare_gauntlet.data import fetch

    monkeypatch.setattr(config, "tushare_pro", lambda **kwargs: object())

    def fake_fetch(pro, endpoint, code, cache_dir, force=False):
        if endpoint == "fina_audit":
            raise OSError("down")
        return pd.DataFrame([{"end_date": "20251231"}])

    monkeypatch.setattr(fetch, "fetch_symbol_table", fake_fetch)
    result = svc.refresh_financials(["600875.SH"], root=tmp_path)
    per = result["codes"]["600875.SH"]
    assert result["ok"] is False
    assert per["fina_audit"]["ok"] is False
    assert "OSError: down" in per["fina_audit"]["error"]
    assert per["income"]["ok"] is True
    assert len(per) == 12


def test_governance_check_reads_complete_local_cache_offline(monkeypatch, tmp_path: Path) -> None:
    from ashare_gauntlet import config
    from ashare_gauntlet.data import fetch

    _write_governance_cache(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline governance attempted external I/O")

    monkeypatch.setattr(svc, "run_module", forbidden)
    monkeypatch.setattr(config, "tushare_pro", forbidden)
    monkeypatch.setattr(fetch, "fetch_symbol_table", forbidden)
    result = svc.governance_check(["600875.sh"], root=tmp_path)
    assert result["ok"] is True
    assert result["data_status"] == "complete"
    assert result["network_access"] is False
    assert result["source"] == "local_parquet"
    per = result["codes"]["600875.SH"]
    assert per["pledge"]["status"] == "found"
    assert per["pledge"]["controller_pledge"]["pledged_ratio_of_holding"] == 40.0
    assert per["audit"]["opinion"]["is_nonstandard"] is False
    assert result["coverage"]["covered_tables"] == 2


@pytest.mark.parametrize(
    ("missing_endpoint", "expected_covered"),
    [("pledge_detail", 1), ("fina_audit", 1), ("both", 0)],
)
def test_governance_check_missing_cache_is_uncovered_without_writes(
    monkeypatch, tmp_path: Path, missing_endpoint: str, expected_covered: int,
) -> None:
    from ashare_gauntlet import config
    from ashare_gauntlet.data import fetch

    pledge, audit = _governance_frames()
    if missing_endpoint != "both":
        endpoint = "fina_audit" if missing_endpoint == "pledge_detail" else "pledge_detail"
        frame = audit if endpoint == "fina_audit" else pledge
        path = tmp_path / "data/cache" / endpoint / "600875.SH.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    def forbidden(*args, **kwargs):
        raise AssertionError("offline governance attempted external I/O")

    monkeypatch.setattr(svc, "run_module", forbidden)
    monkeypatch.setattr(config, "tushare_pro", forbidden)
    monkeypatch.setattr(fetch, "fetch_symbol_table", forbidden)
    result = svc.governance_check(["600875.SH"], root=tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert result["ok"] is False
    assert result["data_status"] == "incomplete"
    assert result["coverage"]["covered_tables"] == expected_covered
    per = result["codes"]["600875.SH"]
    missing = ("pledge_detail", "fina_audit") if missing_endpoint == "both" else (missing_endpoint,)
    for endpoint in missing:
        key = "pledge" if endpoint == "pledge_detail" else "audit"
        assert per[key]["coverage"]["reason"] == "CACHE_MISSING"
        assert per[key]["status"] == "uncovered"
    assert per["warnings"] == ["UNCOVERED_DOES_NOT_MEAN_LOW_RISK"]


@pytest.mark.parametrize("endpoint", ["pledge_detail", "fina_audit"])
def test_governance_check_empty_table_is_uncovered(tmp_path: Path, endpoint: str) -> None:
    pledge, audit = _governance_frames()
    empty = pd.DataFrame(columns=pledge.columns if endpoint == "pledge_detail" else audit.columns)
    _write_governance_cache(
        tmp_path,
        pledge=empty if endpoint == "pledge_detail" else pledge,
        audit=empty if endpoint == "fina_audit" else audit,
    )
    result = svc.governance_check(["600875.SH"], root=tmp_path)
    key = "pledge" if endpoint == "pledge_detail" else "audit"
    assert result["ok"] is False
    assert result["codes"]["600875.SH"][key]["coverage"]["reason"] == "EMPTY_TABLE"
    assert result["codes"]["600875.SH"][key]["status"] == "uncovered"


@pytest.mark.parametrize(
    ("endpoint", "column"),
    [("pledge_detail", "holder_name"), ("fina_audit", "audit_result")],
)
def test_governance_check_incomplete_schema_is_uncovered(
    tmp_path: Path, endpoint: str, column: str,
) -> None:
    pledge, audit = _governance_frames()
    if endpoint == "pledge_detail":
        pledge = pledge.drop(columns=[column])
    else:
        audit = audit.drop(columns=[column])
    _write_governance_cache(tmp_path, pledge=pledge, audit=audit)
    result = svc.governance_check(["600875.SH"], root=tmp_path)
    key = "pledge" if endpoint == "pledge_detail" else "audit"
    coverage = result["codes"]["600875.SH"][key]["coverage"]
    assert coverage["reason"] == "SCHEMA_INCOMPLETE"
    assert coverage["missing_columns"] == [column]


def test_governance_check_surfaces_cache_read_error(monkeypatch, tmp_path: Path) -> None:
    _write_governance_cache(tmp_path)
    real_read = pd.read_parquet

    def fake_read(path, *args, **kwargs):
        if "pledge_detail" in str(path):
            raise OSError("corrupt")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    result = svc.governance_check(["600875.SH"], root=tmp_path)
    per = result["codes"]["600875.SH"]
    assert per["pledge"]["coverage"]["reason"] == "CACHE_READ_ERROR"
    assert "OSError: corrupt" in per["pledge"]["coverage"]["error"]
    assert per["audit"]["status"] == "found"


def test_governance_check_distinguishes_fully_released_and_nonstandard_audit(tmp_path: Path) -> None:
    pledge, audit = _governance_frames()
    pledge.loc[0, "pledged_amount"] = 0.0
    audit.loc[0, "audit_result"] = "保留意见"
    _write_governance_cache(tmp_path, pledge=pledge, audit=audit)
    result = svc.governance_check(["600875.SH"], root=tmp_path)
    per = result["codes"]["600875.SH"]
    assert result["ok"] is True
    assert per["pledge"]["status"] == "none_current"
    assert per["pledge"]["controller_pledge"] is None
    assert per["audit"]["opinion"]["is_nonstandard"] is True


def test_governance_check_isolates_multiple_codes(tmp_path: Path) -> None:
    _write_governance_cache(tmp_path, "600875.SH")
    pledge, _ = _governance_frames()
    path = tmp_path / "data/cache/pledge_detail/600011.SH.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pledge.to_parquet(path, index=False)
    result = svc.governance_check(["600875.SH", "600011.SH"], root=tmp_path)
    assert result["ok"] is False
    assert result["coverage"]["complete_codes"] == 1
    assert result["coverage"]["incomplete_codes"] == ["600011.SH"]
    assert result["codes"]["600875.SH"]["data_status"] == "complete"


def test_governance_check_rejects_force_before_any_io(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "is_file", lambda self: pytest.fail("filesystem inspected"))
    monkeypatch.setattr(svc, "run_module", lambda *args, **kwargs: pytest.fail("subprocess used"))
    with pytest.raises(ValueError, match="refresh_stock_financials"):
        svc.governance_check(["600875.SH"], force=True, root=tmp_path)


# ── P0-1: account_snapshot 统一 schema ──

def test_account_snapshot_returns_unified_schema(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260807", "cash": 1000, "conditional_orders": None,
        "positions": [
            {"ts_code": "000001.SZ", "mv": 9000, "industry": "银行", "bucket": "长线"},
        ],
    })
    result = svc.account_snapshot(tmp_path)
    assert result["schema"] == "account_state.v1"
    assert result["source_schema"] == "legacy_unversioned"
    assert result["total_assets"] == 10000
    # raw 不暴露
    assert "raw" not in result["conditional_orders"]


def test_account_snapshot_legacy_orders_no_raw(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260807", "cash": 1000,
        "conditional_orders": "2张待核对",
        "positions": [],
    })
    result = svc.account_snapshot(tmp_path)
    co = result["conditional_orders"]
    assert co["status"] == "unverified"
    assert co["format"] == "legacy_free_text"
    assert "raw" not in co
    assert co["raw_present"] is True


def test_account_snapshot_v2_orders_verified(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260807", "cash": 1000,
        "conditional_orders": {
            "schema_version": 2,
            "orders": [{
                "order_id": "ord-001",
                "ts_code": "000001.SZ",
                "side": "BUY",
                "condition": {"field": "close", "operator": "<="},
                "price": 10.5,
                "shares": 100,
                "valid_from": "20260801",
                "valid_until": "20260831",
                "status": "active",
            }],
        },
        "positions": [],
    })
    result = svc.account_snapshot(tmp_path)
    co = result["conditional_orders"]
    assert co["status"] == "verified"
    assert co["verified_count"] == 1
    assert "raw" not in co


def test_account_snapshot_v2_orders_invalid(tmp_path: Path) -> None:
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260807", "cash": 1000,
        "conditional_orders": {
            "schema_version": 2,
            "orders": [{
                "order_id": "ord-bad",
                "ts_code": "bad_code",
                "side": "HOLD",
                "condition": {"field": "open", "operator": ">"},
                "price": -1,
                "shares": 0,
                "valid_from": "20260831",
                "valid_until": "20260801",
                "status": "triggered",
            }],
        },
        "positions": [],
    })
    result = svc.account_snapshot(tmp_path)
    co = result["conditional_orders"]
    assert co["status"] == "invalid"
    assert len(co.get("invalid_fields", [])) > 0


def test_readiness_strict_freshness_blocker(tmp_path: Path) -> None:
    """strict freshness: ACCOUNT_AS_OF_FUTURE 同样 block。"""
    for endpoint in ("daily", "adj_factor", "daily_basic", "stk_limit"):
        path = tmp_path / "data/cache" / endpoint / "20260807.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    dump(tmp_path / "data/holdscore/20260807_factor.json", [])
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", {
        "as_of": "20260807", "data_status": "complete",
        "c2_state": _c2_not_initialized(), "decisions": [],
    })
    # 未来日期
    dump(tmp_path / "data/holdings.json", {
        "as_of": "20260809", "cash": 0, "positions": [],
    })
    dump(tmp_path / "data/trading_policy.json", {})
    dump(tmp_path / "data/profile.json", {})
    dump(tmp_path / "data/factcheck_overrides.json", {"overrides": []})
    readiness = svc._recommendation_readiness(tmp_path)
    # 未来日期即使大于 decision as_of 也不是"安全"的
    assert not readiness["ready"]
    assert any("FUTURE" in b or "NOT_ALIGNED" in b for b in readiness["blockers"])


_INVALID_CONDITIONAL_ORDERS_V2 = {
    "schema_version": 2,
    "orders": [{"order_id": "x", "ts_code": "bad", "side": "HOLD",
                "condition": {}, "price": -1, "shares": 0,
                "valid_from": "x", "valid_until": "y", "status": "x"}],
}


@pytest.mark.parametrize(
    ("conditional_orders", "expected_status"),
    [
        (None, "missing"),
        ("待核对", "unverified"),
        (_INVALID_CONDITIONAL_ORDERS_V2, "invalid"),
    ],
)
def test_readiness_ignores_missing_invalid_legacy_conditional_orders(
    tmp_path: Path, conditional_orders: object, expected_status: str,
) -> None:
    """缺失/invalid/legacy 条件单不影响 recommendation readiness。"""
    for endpoint in ("daily", "adj_factor", "daily_basic", "stk_limit"):
        path = tmp_path / "data/cache" / endpoint / "20260807.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    dump(tmp_path / "data/holdscore/20260807_factor.json", [])
    dump(tmp_path / "data/decisions/20260807_buy_decisions.json", {
        "as_of": "20260807", "data_status": "complete",
        "c2_state": _c2_not_initialized(), "decisions": [],
    })
    holdings: dict = {"as_of": "20260807", "cash": 0, "positions": []}
    if conditional_orders is not None:
        holdings["conditional_orders"] = conditional_orders
    dump(tmp_path / "data/holdings.json", holdings)
    dump(tmp_path / "data/trading_policy.json", {})
    dump(tmp_path / "data/profile.json", {})
    dump(tmp_path / "data/factcheck_overrides.json", {"overrides": []})
    readiness = svc._recommendation_readiness(tmp_path)
    assert readiness["ready"] is True
    assert readiness["blockers"] == []
    assert "CONDITIONAL_ORDERS_INVALID" not in readiness["blockers"]
    assert "CONDITIONAL_ORDERS_UNVERIFIED" not in readiness["blockers"]
    assert readiness["components"]["conditional_orders"]["status"] == expected_status
    assert expected_status != "verified"


def test_generate_daily_analysis_supports_new_account_shape(tmp_path: Path, monkeypatch) -> None:
    """现有 generate_daily_analysis 兼容 normalize_account_state 输出。"""
    monkeypatch.setattr(svc, "run_module", lambda module, args, timeout, root: {
        "ok": True, "module": module, "returncode": 0, "stdout": "", "stderr": "",
    })
    monkeypatch.setattr(svc, "latest_decisions", lambda *args, **kwargs: {"summary": {}})
    monkeypatch.setattr(svc, "account_snapshot", lambda root: {
        "data_status": "complete", "as_of": "20260807", "total_assets": 1,
        "short_slot": {"violation": False}, "conditional_orders": {"status": "verified"},
        "schema": "account_state.v1",
    })
    monkeypatch.setattr(svc, "_recommendation_readiness", lambda root: {"ready": True})
    result = svc.generate_daily_analysis(root=tmp_path)
    assert result["ok"] is True
