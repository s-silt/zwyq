from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.c2_review import main


CORE_COLUMNS = {
    "daily": ["ts_code", "trade_date", "open", "high", "low", "close"],
    "adj_factor": ["ts_code", "trade_date", "adj_factor"],
    "daily_basic": [
        "ts_code", "trade_date", "total_mv", "pe_ttm", "pb", "turnover_rate",
    ],
    "stk_limit": ["ts_code", "trade_date", "up_limit", "down_limit"],
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_calendar(root: Path, as_of: str) -> Path:
    month_start = pd.Timestamp(f"{as_of[:4]}-{as_of[4:6]}-01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    days = pd.date_range(month_start, month_end)
    frame = pd.DataFrame({
        "cal_date": days.strftime("%Y%m%d"),
        "is_open": [int(day.weekday() < 5) for day in days],
    })
    path = root / "data/cache/trade_cal" / f"{as_of[:6]}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _write_core(root: Path, as_of: str) -> None:
    values = {
        "daily": ["A", as_of, 10.0, 11.0, 9.0, 10.5],
        "adj_factor": ["A", as_of, 1.0],
        "daily_basic": ["A", as_of, 1000.0, 10.0, 1.0, 2.0],
        "stk_limit": ["A", as_of, 11.5, 9.5],
    }
    for endpoint, columns in CORE_COLUMNS.items():
        path = root / "data/cache" / endpoint / f"{as_of}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([values[endpoint]], columns=columns).to_parquet(path, index=False)


def write_valid_review_fixture(
    root: Path,
    *,
    as_of: str = "20260130",
    outside: set[str] | None = None,
    decisions: list[dict] | None = None,
    factor_rows: object | None = None,
) -> Path:
    outside = outside or set()
    _write_calendar(root, as_of)
    _write_core(root, as_of)
    if decisions is None:
        decisions = [
            {
                "ts_code": "A",
                "name": "Alpha",
                "state": "HOLD",
                "reason_codes": ["HELD"],
                "eligible_buy": False,
                "industry": "Synthetic",
            }
        ]
    if factor_rows is None:
        factor_rows = [
            {"ts_code": "A", "name": "Alpha", "decile": 9 if "A" in outside else 10}
        ]
    factor = root / "data/holdscore" / f"{as_of}_factor.json"
    _write_json(factor, factor_rows)
    decision = root / "data/decisions" / f"{as_of}_buy_decisions.json"
    _write_json(decision, {
        "as_of": as_of,
        "factor_snapshot": factor.relative_to(root).as_posix(),
        "decisions": decisions,
    })
    return decision


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return int(exc.value.code)


def _last_summary(capsys: pytest.CaptureFixture[str]) -> dict:
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines
    return json.loads(lines[-1])


def test_month_end_review_hashes_sources_and_advances_state(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path, as_of="20260130", outside={"A"})
    state_path = tmp_path / "data/decisions/c2_review_state.json"
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(tmp_path), "--decision", str(decision), "--state", str(state_path)])
    assert exc.value.code == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    review = state["reviews"][-1]
    assert review["status"] == "VALID"
    assert review["decision_snapshot"]["sha256"] == hashlib.sha256(decision.read_bytes()).hexdigest()
    assert review["decision_snapshot"]["path"] == "data/decisions/20260130_buy_decisions.json"
    assert review["factor_snapshot"]["path"] == "data/holdscore/20260130_factor.json"
    assert state["positions"]["A"]["status"] == "WATCH"
    assert "+" in state["updated_at"]


def test_second_valid_outside_review_exits_two_and_reports_eligible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    first = write_valid_review_fixture(tmp_path, as_of="20260130", outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(first)]) == 0
    second = write_valid_review_fixture(tmp_path, as_of="20260227", outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(second)]) == 2
    summary = _last_summary(capsys)
    assert summary["status"] == "VALID"
    assert summary["newly_exit_eligible"] == ["A"]


def test_ordinary_trading_day_is_not_due_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    decision = write_valid_review_fixture(tmp_path, as_of="20260129")
    state = tmp_path / "data/decisions/c2_review_state.json"
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 0
    assert not state.exists()
    assert _last_summary(capsys)["status"] == "NOT_DUE"


def test_calendar_must_prove_coverage_through_natural_month_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    decision = write_valid_review_fixture(tmp_path)
    calendar = tmp_path / "data/cache/trade_cal/202601.parquet"
    frame = pd.read_parquet(calendar)
    frame[frame["cal_date"].astype(str) != "20260131"].to_parquet(calendar, index=False)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1
    assert _last_summary(capsys)["status"] == "REVIEW_BLOCKED_DATA"
    assert not (tmp_path / "data/decisions/c2_review_state.json").exists()


def test_calendar_rejects_non_binary_is_open(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    calendar = tmp_path / "data/cache/trade_cal/202601.parquet"
    frame = pd.read_parquet(calendar)
    frame["is_open"] = frame["is_open"].astype(float)
    frame.loc[frame.index[0], "is_open"] = 0.5
    frame.to_parquet(calendar, index=False)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_impossible_decision_date_is_rejected(tmp_path: Path) -> None:
    decision = tmp_path / "data/decisions/20260230_buy_decisions.json"
    _write_json(decision, {"as_of": "20260230", "factor_snapshot": "x", "decisions": []})
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_malformed_later_decision_records_block_when_filename_identifies_period(
    tmp_path: Path,
) -> None:
    first = write_valid_review_fixture(tmp_path, as_of="20260130", outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(first)]) == 0
    second = write_valid_review_fixture(tmp_path, as_of="20260227", outside={"A"})
    second.write_bytes(b"{malformed")
    assert _run(["--root", str(tmp_path), "--decision", str(second)]) == 1
    state = json.loads(
        (tmp_path / "data/decisions/c2_review_state.json").read_text("utf-8")
    )
    assert state["reviews"][-1]["status"] == "REVIEW_BLOCKED_DATA"
    assert state["positions"]["A"]["out_streak"] == 1


@pytest.mark.parametrize("endpoint", sorted(CORE_COLUMNS))
def test_each_core_endpoint_is_required(tmp_path: Path, endpoint: str) -> None:
    decision = write_valid_review_fixture(tmp_path)
    (tmp_path / "data/cache" / endpoint / "20260130.parquet").unlink()
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_empty_core_partition_is_rejected(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    path = tmp_path / "data/cache/daily/20260130.parquet"
    pd.DataFrame(columns=CORE_COLUMNS["daily"]).to_parquet(path, index=False)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_date_misaligned_core_partition_is_rejected(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    path = tmp_path / "data/cache/adj_factor/20260130.parquet"
    frame = pd.read_parquet(path)
    frame["trade_date"] = "20260129"
    frame.to_parquet(path, index=False)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_daily_basic_requires_all_production_fields(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    path = tmp_path / "data/cache/daily_basic/20260130.parquet"
    pd.read_parquet(path).drop(columns=["turnover_rate"]).to_parquet(path, index=False)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_core_numeric_fields_reject_non_numeric_text(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    path = tmp_path / "data/cache/daily/20260130.parquet"
    frame = pd.read_parquet(path)
    frame["close"] = "not-a-number"
    frame.to_parquet(path, index=False)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


@pytest.mark.parametrize("rows", [
    {},
    [],
    [{"ts_code": "A", "decile": True}],
    [{"ts_code": "A", "decile": 0}],
    [{"ts_code": "", "decile": 10}],
    [{"ts_code": "A"}],
])
def test_malformed_factor_rows_are_rejected(tmp_path: Path, rows: object) -> None:
    decision = write_valid_review_fixture(tmp_path, factor_rows=rows)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_duplicate_factor_ts_code_is_rejected(tmp_path: Path) -> None:
    rows = [{"ts_code": "A", "decile": 10}, {"ts_code": "A", "decile": 9}]
    decision = write_valid_review_fixture(tmp_path, factor_rows=rows)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_decision_as_of_must_match_dated_filename(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["as_of"] = "20260129"
    _write_json(decision, payload)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_referenced_factor_path_cannot_escape_root(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["factor_snapshot"] = "../outside_factor.json"
    _write_json(decision, payload)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_referenced_factor_filename_date_must_match_decision(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    old_factor = tmp_path / "data/holdscore/20260129_factor.json"
    old_factor.write_bytes((tmp_path / "data/holdscore/20260130_factor.json").read_bytes())
    payload = json.loads(decision.read_text("utf-8"))
    payload["factor_snapshot"] = old_factor.relative_to(tmp_path).as_posix()
    _write_json(decision, payload)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1


def test_state_output_cannot_equal_source_path(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    before = decision.read_bytes()
    assert _run([
        "--root", str(tmp_path), "--decision", str(decision), "--state", str(decision),
    ]) == 1
    assert decision.read_bytes() == before


def test_corrupt_existing_state_is_preserved_byte_for_byte(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    state = tmp_path / "data/decisions/c2_review_state.json"
    corrupt = b"{definitely not json\r\n"
    state.write_bytes(corrupt)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1
    assert state.read_bytes() == corrupt


def test_atomic_write_failure_preserves_old_state_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = write_valid_review_fixture(tmp_path, as_of="20260130", outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(first)]) == 0
    state = tmp_path / "data/decisions/c2_review_state.json"
    before = state.read_bytes()
    second = write_valid_review_fixture(tmp_path, as_of="20260227", outside={"A"})

    import scripts.c2_review as cli

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(cli.os, "replace", fail_replace)
    assert _run(["--root", str(tmp_path), "--decision", str(second)]) == 1
    assert state.read_bytes() == before
    assert list(state.parent.glob(".tmp_c2_review_*")) == []


def test_atomic_write_cleans_temp_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.c2_review as cli

    destination = tmp_path / "state.json"

    def fail_fdopen(*_args: object, **_kwargs: object):
        raise OSError("synthetic fdopen failure")

    monkeypatch.setattr(cli.os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="synthetic"):
        cli._atomic_write_json(destination, {"ok": True})
    assert list(tmp_path.glob(".tmp_c2_review_*")) == []


def test_same_period_replay_is_idempotent_and_does_not_rewrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    decision = write_valid_review_fixture(tmp_path, outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 0
    state = tmp_path / "data/decisions/c2_review_state.json"
    before = state.read_bytes()
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 0
    assert state.read_bytes() == before
    assert _last_summary(capsys)["status"] == "IDEMPOTENT"


def test_same_period_identical_hashes_are_idempotent_without_calendar(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path, outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 0
    state = tmp_path / "data/decisions/c2_review_state.json"
    before = state.read_bytes()
    (tmp_path / "data/cache/trade_cal/202601.parquet").unlink()
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 0
    assert state.read_bytes() == before


@pytest.mark.parametrize("changed_source", ["decision", "factor"])
def test_same_period_changed_source_hash_is_rejected(
    tmp_path: Path, changed_source: str,
) -> None:
    decision = write_valid_review_fixture(tmp_path, outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 0
    state = tmp_path / "data/decisions/c2_review_state.json"
    before = state.read_bytes()
    target = decision if changed_source == "decision" else tmp_path / "data/holdscore/20260130_factor.json"
    target.write_bytes(target.read_bytes() + b"\n")
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1
    assert state.read_bytes() == before


def test_changed_same_period_decision_conflicts_before_invalid_factor_can_record_block(
    tmp_path: Path,
) -> None:
    decision = write_valid_review_fixture(tmp_path, outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 0
    state = tmp_path / "data/decisions/c2_review_state.json"
    before = state.read_bytes()
    payload = json.loads(decision.read_text("utf-8"))
    payload["factor_snapshot"] = "../escaped.json"
    _write_json(decision, payload)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1
    assert state.read_bytes() == before


def test_raw_sources_are_hashed_before_json_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = write_valid_review_fixture(tmp_path)
    import scripts.c2_review as cli

    events: list[str] = []
    source_bytes = {
        decision.read_bytes(),
        (tmp_path / "data/holdscore/20260130_factor.json").read_bytes(),
    }
    real_sha256 = cli.hashlib.sha256
    real_loads = cli.json.loads

    def tracked_sha256(value: bytes):
        if value in source_bytes:
            events.append("hash")
        return real_sha256(value)

    def tracked_loads(value: object, *args: object, **kwargs: object):
        comparable = value.encode("utf-8") if isinstance(value, str) else value
        if comparable in source_bytes:
            events.append("parse")
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(cli.hashlib, "sha256", tracked_sha256)
    monkeypatch.setattr(cli.json, "loads", tracked_loads)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 0
    assert events[:4] == ["hash", "parse", "hash", "parse"]


@pytest.mark.parametrize("reason", [
    "GOVERNANCE_RED", "RISK_LINE_BREACH", "MANUAL_LOGIC_FAIL",
])
def test_immediate_exit_reasons_bypass_factor_and_clear_watch(
    tmp_path: Path, reason: str,
) -> None:
    first = write_valid_review_fixture(tmp_path, as_of="20260130", outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(first)]) == 0
    exits = [{"ts_code": "A", "name": "Alpha", "state": "EXIT", "reason_codes": [reason]}]
    second = write_valid_review_fixture(
        tmp_path, as_of="20260227", outside={"A"}, decisions=exits,
    )
    assert _run(["--root", str(tmp_path), "--decision", str(second)]) == 0
    state = json.loads((tmp_path / "data/decisions/c2_review_state.json").read_text("utf-8"))
    assert "A" not in state["positions"]
    assert state["reviews"][-1]["transitions"][0]["action"] == "BYPASS"


@pytest.mark.parametrize(("decile", "expected_code", "expected_status"), [
    (10, 0, None),
    (9, 2, "EXIT_ELIGIBLE"),
])
def test_c2_derived_exit_is_decided_independently_from_factor_d10(
    tmp_path: Path, decile: int, expected_code: int, expected_status: str | None,
) -> None:
    first = write_valid_review_fixture(tmp_path, as_of="20260130", outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(first)]) == 0
    exits = [{
        "ts_code": "A", "name": "Alpha", "state": "EXIT",
        "reason_codes": ["EXIT_RULE_C2_CONFIRMED"], "eligible_buy": True,
        "tier": "green", "industry": "Anything",
    }]
    second = write_valid_review_fixture(
        tmp_path,
        as_of="20260227",
        decisions=exits,
        factor_rows=[{"ts_code": "A", "decile": decile}],
    )
    assert _run(["--root", str(tmp_path), "--decision", str(second)]) == expected_code
    state = json.loads((tmp_path / "data/decisions/c2_review_state.json").read_text("utf-8"))
    assert state["positions"].get("A", {}).get("status") == expected_status


@pytest.mark.parametrize("decisions", [
    [
        {"ts_code": "A", "name": "Alpha", "state": "HOLD", "reason_codes": ["HELD"]},
        {"ts_code": "A", "name": "Alpha", "state": "HOLD", "reason_codes": ["HELD"]},
    ],
    [
        {"ts_code": "A", "name": "Alpha", "state": "HOLD", "reason_codes": ["HELD"]},
        {"ts_code": "A", "name": "Alpha", "state": "EXIT", "reason_codes": ["RISK_LINE_BREACH"]},
    ],
])
def test_duplicate_or_contradictory_held_decisions_block_whole_review(
    tmp_path: Path, decisions: list[dict],
) -> None:
    decision = write_valid_review_fixture(tmp_path, decisions=decisions)
    assert _run(["--root", str(tmp_path), "--decision", str(decision)]) == 1
    assert not (tmp_path / "data/decisions/c2_review_state.json").exists()


def test_blocked_review_is_appended_once_without_changing_streak(tmp_path: Path) -> None:
    first = write_valid_review_fixture(tmp_path, as_of="20260130", outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(first)]) == 0
    second = write_valid_review_fixture(tmp_path, as_of="20260227", outside={"A"})
    (tmp_path / "data/cache/daily/20260227.parquet").unlink()
    assert _run(["--root", str(tmp_path), "--decision", str(second)]) == 1
    state_path = tmp_path / "data/decisions/c2_review_state.json"
    once = json.loads(state_path.read_text("utf-8"))
    assert once["positions"]["A"]["out_streak"] == 1
    assert once["reviews"][-1]["status"] == "REVIEW_BLOCKED_DATA"
    assert "daily/20260227" in " ".join(once["reviews"][-1]["issues"])
    before_replay = state_path.read_bytes()
    assert _run(["--root", str(tmp_path), "--decision", str(second)]) == 1
    assert state_path.read_bytes() == before_replay


def test_malformed_factor_block_records_available_raw_hash(tmp_path: Path) -> None:
    first = write_valid_review_fixture(tmp_path, as_of="20260130", outside={"A"})
    assert _run(["--root", str(tmp_path), "--decision", str(first)]) == 0
    second = write_valid_review_fixture(tmp_path, as_of="20260227", outside={"A"})
    factor = tmp_path / "data/holdscore/20260227_factor.json"
    factor.write_bytes(b"[malformed")
    expected_hash = hashlib.sha256(factor.read_bytes()).hexdigest()
    assert _run(["--root", str(tmp_path), "--decision", str(second)]) == 1
    state = json.loads(
        (tmp_path / "data/decisions/c2_review_state.json").read_text("utf-8")
    )
    assert state["reviews"][-1]["evidence_hashes"]["factor_snapshot"] == expected_hash


def test_omitted_decision_discovers_latest_snapshot(tmp_path: Path) -> None:
    write_valid_review_fixture(tmp_path, outside={"A"})
    assert _run(["--root", str(tmp_path)]) == 0
    assert (tmp_path / "data/decisions/c2_review_state.json").exists()


def test_relative_explicit_paths_resolve_under_root(tmp_path: Path) -> None:
    write_valid_review_fixture(tmp_path, outside={"A"})
    assert _run([
        "--root", str(tmp_path),
        "--decision", "data/decisions/20260130_buy_decisions.json",
        "--state", "data/decisions/custom_c2_state.json",
    ]) == 0
    assert (tmp_path / "data/decisions/custom_c2_state.json").exists()


def test_paths_outside_root_are_rejected(tmp_path: Path) -> None:
    decision = write_valid_review_fixture(tmp_path)
    outside = tmp_path.parent / "outside_c2_state.json"
    assert _run([
        "--root", str(tmp_path), "--decision", str(decision), "--state", str(outside),
    ]) == 1
    assert not outside.exists()


def test_help_exits_zero() -> None:
    assert _run(["--help"]) == 0


@pytest.mark.parametrize("argv", [["--unknown"], ["--root"]])
def test_invalid_or_unknown_arguments_exit_one(argv: list[str]) -> None:
    assert _run(argv) == 1
