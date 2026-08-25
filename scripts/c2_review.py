"""Validate evidence and persist the production C2 monthly review sidecar."""
from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import pandas as pd

from ashare_gauntlet.c2_review import (
    C2ReviewError,
    advance_review,
    eligible_codes,
    initial_state,
    record_blocked_review,
    validate_state,
)


CORE_COLUMNS = {
    "daily": {"ts_code", "trade_date", "open", "high", "low", "close"},
    "adj_factor": {"ts_code", "trade_date", "adj_factor"},
    "daily_basic": {
        "ts_code", "trade_date", "total_mv", "pe_ttm", "pb", "turnover_rate",
    },
    "stk_limit": {"ts_code", "trade_date", "up_limit", "down_limit"},
}

_DECISION_RE = re.compile(r"^(\d{8})_buy_decisions\.json$")
_IMMEDIATE_BYPASS_REASONS = frozenset({
    "GOVERNANCE_RED", "RISK_LINE_BREACH", "MANUAL_LOGIC_FAIL",
})


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise C2ReviewError(f"invalid arguments: {message}")


class _ReviewConflict(C2ReviewError):
    """A frozen valid period was presented with different raw evidence."""


def _inside_root(root: Path, path: Path) -> Path:
    resolved_root, resolved = root.resolve(), path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise C2ReviewError(f"path escapes root: {path}")
    return resolved


def _resolve_under_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return _inside_root(root, path if path.is_absolute() else root / path)


def _validate_state_target(root: Path, state_path: Path) -> None:
    relative = state_path.relative_to(root.resolve())
    parts = tuple(part.casefold() for part in relative.parts)
    fixed = {
        ("data", "holdings.json"),
        ("data", "profile.json"),
        ("data", "trading_policy.json"),
    }
    protected_namespace = len(parts) >= 2 and parts[:2] in {
        ("data", "cache"), ("data", "holdscore"),
    }
    if parts in fixed or protected_namespace or _DECISION_RE.fullmatch(state_path.name):
        raise C2ReviewError(f"protected state output target: {relative.as_posix()}")


def _real_date(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise C2ReviewError(f"{label} must be a real YYYYMMDD date")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise C2ReviewError(f"{label} must be a real YYYYMMDD date") from exc
    return value


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise C2ReviewError(f"{label} is unreadable") from exc


def _parse_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2ReviewError(f"{label} is not valid UTF-8 JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root.resolve()).as_posix()


def _is_month_end(root: Path, as_of: str) -> bool:
    """Require cached calendar through calendar month end, then compare last open day."""
    checked = _real_date(as_of, "as_of")
    year, month = int(checked[:4]), int(checked[4:6])
    first = f"{checked[:6]}01"
    final_day = calendar.monthrange(year, month)[1]
    final = f"{checked[:6]}{final_day:02d}"
    directory = _inside_root(root, root / "data/cache/trade_cal")
    paths = sorted(directory.glob("*.parquet")) if directory.exists() else []
    if not paths:
        raise C2ReviewError("trade_cal cache missing")
    frames: list[pd.DataFrame] = []
    for candidate in paths:
        path = _inside_root(root, candidate)
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise C2ReviewError("trade_cal cache unreadable") from exc
        if not {"cal_date", "is_open"}.issubset(frame.columns):
            raise C2ReviewError("trade_cal cache missing required fields")
        frames.append(frame[["cal_date", "is_open"]])
    calendar_rows = pd.concat(frames, ignore_index=True)
    if calendar_rows.empty:
        raise C2ReviewError("trade_cal cache empty")
    calendar_rows["cal_date"] = calendar_rows["cal_date"].astype(str)
    calendar_rows = calendar_rows[
        (calendar_rows["cal_date"] >= first) & (calendar_rows["cal_date"] <= final)
    ]
    expected = {
        day.strftime("%Y%m%d")
        for day in pd.date_range(datetime(year, month, 1), datetime(year, month, final_day))
    }
    actual = set(calendar_rows["cal_date"])
    if not expected.issubset(actual):
        raise C2ReviewError("trade_cal cache does not prove coverage through month end")
    if calendar_rows["cal_date"].duplicated().any():
        duplicates = calendar_rows[calendar_rows["cal_date"].duplicated(False)]
        for _, group in duplicates.groupby("cal_date"):
            if len(set(pd.to_numeric(group["is_open"], errors="coerce"))) != 1:
                raise C2ReviewError("trade_cal cache has contradictory duplicate rows")
        calendar_rows = calendar_rows.drop_duplicates("cal_date", keep="last")
    open_values = pd.to_numeric(calendar_rows["is_open"], errors="coerce")
    if open_values.isna().any() or not set(open_values).issubset({0, 1}):
        raise C2ReviewError("trade_cal cache has invalid is_open values")
    open_days = sorted(calendar_rows.loc[open_values == 1, "cal_date"])
    if not open_days:
        raise C2ReviewError("trade_cal cache has no open day in review month")
    if checked not in open_days:
        raise C2ReviewError("decision as_of is not an open trading day")
    return checked == open_days[-1]


def _validate_core(cache: Path, as_of: str) -> None:
    """Require each exact partition, non-empty rows, one matching date, and CORE_COLUMNS."""
    root = cache.resolve().parent.parent
    for endpoint, required in CORE_COLUMNS.items():
        path = _inside_root(root, cache / endpoint / f"{as_of}.parquet")
        if not path.is_file():
            raise C2ReviewError(f"core endpoint partition missing: {endpoint}/{as_of}")
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise C2ReviewError(f"core endpoint partition unreadable: {endpoint}/{as_of}") from exc
        missing = required - set(frame.columns)
        if missing:
            raise C2ReviewError(
                f"core endpoint fields missing: {endpoint}/{as_of}: {sorted(missing)}"
            )
        if frame.empty:
            raise C2ReviewError(f"core endpoint partition empty: {endpoint}/{as_of}")
        if set(frame["trade_date"].astype(str)) != {as_of}:
            raise C2ReviewError(f"core endpoint partition date mismatch: {endpoint}/{as_of}")
        codes = frame["ts_code"]
        if codes.isna().any() or any(not str(code) for code in codes):
            raise C2ReviewError(f"core endpoint has invalid ts_code: {endpoint}/{as_of}")
        if codes.astype(str).duplicated().any():
            raise C2ReviewError(f"core endpoint has duplicate ts_code: {endpoint}/{as_of}")
        numeric_columns = required - {"ts_code", "trade_date"}
        converted: dict[str, pd.Series] = {}
        for column in numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & (
                values.isna() | ~values.map(math.isfinite)
            )
            if bool(invalid.any()):
                raise C2ReviewError(
                    f"core endpoint has invalid numeric field {column}: {endpoint}/{as_of}"
                )
            converted[column] = values
        if endpoint == "daily":
            present = pd.DataFrame(converted)[["open", "high", "low", "close"]].notna().sum(axis=1)
            if bool(((present != 0) & (present != 4)).any()):
                raise C2ReviewError(f"daily OHLC must be all present or all null: {as_of}")
            traded = present == 4
            daily = pd.DataFrame(converted)
            invalid_ohlc = traded & (
                (daily[["open", "high", "low", "close"]] <= 0).any(axis=1)
                | (daily["low"] > daily["high"])
                | (daily["open"] < daily["low"])
                | (daily["open"] > daily["high"])
                | (daily["close"] < daily["low"])
                | (daily["close"] > daily["high"])
            )
            if bool(invalid_ohlc.any()):
                raise C2ReviewError(f"daily OHLC values are inconsistent: {as_of}")
        elif endpoint in {"adj_factor", "stk_limit"}:
            values = pd.DataFrame(converted)
            if bool((values.isna() | (values <= 0)).any(axis=None)):
                raise C2ReviewError(f"core endpoint has missing or non-positive values: {endpoint}/{as_of}")


def _validate_factor_rows(value: Any) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise C2ReviewError("factor snapshot must be a non-empty list")
    rows: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise C2ReviewError(f"factor row {index} must be an object")
        code = row.get("ts_code")
        decile = row.get("decile")
        if not isinstance(code, str) or not code:
            raise C2ReviewError(f"factor row {index} has invalid ts_code")
        if code in seen:
            raise C2ReviewError(f"factor snapshot has duplicate ts_code {code}")
        if not isinstance(decile, int) or isinstance(decile, bool) or not 1 <= decile <= 10:
            raise C2ReviewError(f"factor row {index} has invalid decile")
        seen.add(code)
        rows.append(row)
    return rows


def _observations(decision: dict, factor_rows: list[dict]) -> list[dict]:
    """Use HOLD/EXIT for held set and map D10 independently from factor rows."""
    rows = decision.get("decisions")
    if not isinstance(rows, list):
        raise C2ReviewError("decision snapshot decisions must be a list")
    factors = {row["ts_code"]: row["decile"] for row in factor_rows}
    held: dict[str, dict] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise C2ReviewError(f"decision row {index} must be an object")
        state = row.get("state")
        if state not in {"BUY", "WAIT", "HOLD", "EXIT"}:
            raise C2ReviewError(f"decision row {index} has invalid state")
        code = row.get("ts_code")
        name = row.get("name")
        if not isinstance(code, str) or not code or not isinstance(name, str) or not name:
            raise C2ReviewError(f"decision row {index} is missing ts_code or name")
        if state not in {"HOLD", "EXIT"}:
            continue
        reasons = row.get("reason_codes")
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise C2ReviewError(f"held decision row {index} has invalid reason_codes")
        if code in held:
            prior = held[code]
            qualifier = "contradictory" if prior["state"] != state else "duplicate"
            raise C2ReviewError(f"decision snapshot has {qualifier} held row {code}")
        held[code] = row

    observations: list[dict] = []
    for code in sorted(held):
        row = held[code]
        reasons = set(row["reason_codes"])
        if row["state"] == "EXIT" and reasons & _IMMEDIATE_BYPASS_REASONS:
            status = "BYPASS"
        else:
            status = "INSIDE" if factors.get(code) == 10 else "OUTSIDE"
        observations.append({"ts_code": code, "name": row["name"], "status": status})
    return observations


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write allow_nan=False with temporary file, flush, fsync, replace, cleanup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp_c2_review_", suffix=".json", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_state(path: Path) -> tuple[dict | None, bool]:
    if not path.exists():
        return None, False
    payload = _parse_json(_read_bytes(path, "existing C2 state"), "existing C2 state")
    if not isinstance(payload, dict):
        raise C2ReviewError("existing C2 state must be an object")
    validate_state(payload)
    return payload, True


def _discover_decision(root: Path) -> Path:
    directory = _inside_root(root, root / "data/decisions")
    matches = sorted(
        path for path in directory.glob("*_buy_decisions.json")
        if _DECISION_RE.fullmatch(path.name)
    ) if directory.exists() else []
    if not matches:
        raise C2ReviewError("no dated buy decision snapshot found")
    return _inside_root(root, matches[-1])


def _factor_source(
    root: Path, state_path: Path, decision: dict, as_of: str,
) -> tuple[Path, bytes, str]:
    factor_reference = decision.get("factor_snapshot")
    if not isinstance(factor_reference, str) or not factor_reference:
        raise C2ReviewError("decision factor_snapshot must be a non-empty path string")
    factor_path = _resolve_under_root(root, factor_reference)
    if not re.search(rf"(?:^|[/\\]){as_of}_factor\.json$", factor_reference):
        raise C2ReviewError("factor_snapshot filename date must match decision as_of")
    if state_path == factor_path:
        raise C2ReviewError("state output path must differ from source input")
    factor_raw = _read_bytes(factor_path, "factor snapshot")
    return factor_path, factor_raw, _sha256(factor_raw)


def _summary_error(message: str, *, status: str = "ERROR", **fields: Any) -> dict:
    return {"status": status, **fields, "error": message}


def _stamp_for_write(state: dict) -> dict:
    stamped = dict(state)
    stamped["updated_at"] = datetime.now(timezone.utc).isoformat()
    validate_state(stamped)
    return stamped


def _blocked_result(
    *,
    state: dict | None,
    state_path: Path,
    period: str,
    as_of: str,
    issue: str,
    evidence_hashes: dict[str, str],
) -> tuple[dict, int]:
    recorded = False
    if state is not None:
        blocked = record_blocked_review(
            state,
            period=period,
            as_of=as_of,
            issues=[issue],
            evidence_hashes=evidence_hashes,
        )
        if blocked != state:
            _atomic_write_json(state_path, _stamp_for_write(blocked))
            recorded = True
    return {
        "status": "REVIEW_BLOCKED_DATA",
        "period": period,
        "as_of": as_of,
        "issues": [issue],
        "recorded": recorded,
    }, 1


def _run(args: argparse.Namespace) -> tuple[dict, int]:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise C2ReviewError("root must be an existing directory")
    decision_path = (
        _resolve_under_root(root, args.decision)
        if args.decision is not None else _discover_decision(root)
    )
    state_path = _resolve_under_root(
        root, args.state or "data/decisions/c2_review_state.json",
    )
    _validate_state_target(root, state_path)
    if state_path == decision_path:
        raise C2ReviewError("state output path must differ from source input")

    match = _DECISION_RE.fullmatch(decision_path.name)
    if match is None:
        raise C2ReviewError("decision filename must be YYYYMMDD_buy_decisions.json")
    file_as_of = _real_date(match.group(1), "decision filename date")
    period = file_as_of[:6]
    decision_raw = _read_bytes(decision_path, "decision snapshot")
    decision_hash = _sha256(decision_raw)

    state, _ = _load_state(state_path)
    existing = next((
        review for review in (state or {}).get("reviews", [])
        if review["status"] == "VALID" and review["period"] == period
    ), None)
    if (existing is not None
            and existing["decision_snapshot"]["sha256"] != decision_hash):
        raise _ReviewConflict(f"valid review conflict for period {period}")

    try:
        decision = _parse_json(decision_raw, "decision snapshot")
        if not isinstance(decision, dict):
            raise C2ReviewError("decision snapshot must be an object")
        as_of = _real_date(decision.get("as_of"), "decision as_of")
        if as_of != file_as_of:
            raise C2ReviewError("decision as_of must match dated decision filename")
    except C2ReviewError as exc:
        return _blocked_result(
            state=state,
            state_path=state_path,
            period=period,
            as_of=file_as_of,
            issue=str(exc),
            evidence_hashes={"decision_snapshot": decision_hash},
        )
    period = as_of[:6]
    if existing is not None:
        try:
            _, _, factor_hash = _factor_source(root, state_path, decision, as_of)
        except C2ReviewError as exc:
            raise _ReviewConflict(
                f"valid review evidence unavailable for period {period}: {exc}"
            ) from exc
        if existing["factor_snapshot"]["sha256"] != factor_hash:
            raise _ReviewConflict(f"valid review conflict for period {period}")
        return {
            "status": "IDEMPOTENT",
            "period": period,
            "as_of": as_of,
            "newly_exit_eligible": [],
            "eligible_codes": sorted(eligible_codes(state)),
        }, 0

    try:
        month_end = _is_month_end(root, as_of)
    except C2ReviewError as exc:
        return _blocked_result(
            state=state,
            state_path=state_path,
            period=period,
            as_of=as_of,
            issue=str(exc),
            evidence_hashes={"decision_snapshot": decision_hash},
        )
    if not month_end:
        return {"status": "NOT_DUE", "period": period, "as_of": as_of}, 0

    factor_hash: str | None = None
    try:
        factor_path, factor_raw, factor_hash = _factor_source(
            root, state_path, decision, as_of,
        )
        factor_value = _parse_json(factor_raw, "factor snapshot")
        factor_rows = _validate_factor_rows(factor_value)
    except C2ReviewError as exc:
        blocked_hashes = {"decision_snapshot": decision_hash}
        if factor_hash is not None:
            blocked_hashes["factor_snapshot"] = factor_hash
        return _blocked_result(
            state=state,
            state_path=state_path,
            period=period,
            as_of=as_of,
            issue=str(exc),
            evidence_hashes=blocked_hashes,
        )

    assert factor_hash is not None
    evidence_hashes = {
        "decision_snapshot": decision_hash,
        "factor_snapshot": factor_hash,
    }
    try:
        _validate_core(_inside_root(root, root / "data/cache"), as_of)
        observations = _observations(decision, factor_rows)
    except C2ReviewError as exc:
        return _blocked_result(
            state=state,
            state_path=state_path,
            period=period,
            as_of=as_of,
            issue=str(exc),
            evidence_hashes=evidence_hashes,
        )

    evidence = {
        "period": period,
        "as_of": as_of,
        "decision_snapshot": {
            "path": _relative_path(root, decision_path), "sha256": decision_hash,
        },
        "factor_snapshot": {
            "path": _relative_path(root, factor_path), "sha256": factor_hash,
        },
        "observations": observations,
    }
    advanced, events = advance_review(state if state is not None else initial_state(), evidence)
    stamped = _stamp_for_write(advanced)
    _atomic_write_json(state_path, stamped)
    newly = events["newly_exit_eligible"]
    return {
        "status": "VALID",
        "period": period,
        "as_of": as_of,
        "observation_count": len(observations),
        "newly_exit_eligible": newly,
        "eligible_codes": sorted(eligible_codes(stamped)),
    }, 2 if newly else 0


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description="Validate and persist a monthly C2 holding review")
    parser.add_argument("--root", default=".")
    parser.add_argument("--decision")
    parser.add_argument("--state")
    return parser


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> None:
    try:
        args = _parser().parse_args(argv)
        summary, code = _run(args)
    except C2ReviewError as exc:
        summary, code = _summary_error(str(exc)), 1
    except Exception as exc:
        summary, code = _summary_error(f"{type(exc).__name__}: operation failed"), 1
    _emit(summary)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
