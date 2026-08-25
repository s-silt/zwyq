"""Validation shared by consumers of persisted decision snapshots."""
from __future__ import annotations

from datetime import datetime
import re
from typing import Any


_C2_FIELDS = {
    "status", "last_valid_review_as_of", "watch", "exit_eligible", "error",
}
_CONSUMABLE_C2_STATUSES = {
    "AVAILABLE", "NOT_INITIALIZED", "REVIEW_BLOCKED_DATA",
}
_DATE8 = re.compile(r"^\d{8}$")
_TS_CODE = re.compile(r"^\d{6}\.(?:SH|SZ)$")
_DECISION_STATES = frozenset({"BUY", "WAIT", "HOLD", "EXIT"})
_BLOCKED_ERROR_PREFIX = "REVIEW_BLOCKED_DATA:"


def _validate_real_date(value: Any, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _DATE8.fullmatch(value):
        raise ValueError(f"{label} must be None or a real YYYYMMDD date")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{label} must be None or a real YYYYMMDD date") from exc


def _validate_code_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not all(isinstance(code, str) and code for code in value):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    if value != sorted(value):
        raise ValueError(f"{label} must be sorted")
    return value


def validate_c2_projection(value: Any, *, label: str = "c2_state") -> dict[str, Any]:
    """Validate the exact projection emitted by ``scripts.buy_list``."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if set(value) != _C2_FIELDS:
        missing = sorted(_C2_FIELDS - set(value))
        extra = sorted(set(value) - _C2_FIELDS)
        raise ValueError(f"{label} fields mismatch: missing={missing} extra={extra}")

    status = value["status"]
    if not isinstance(status, str) or status not in _CONSUMABLE_C2_STATUSES:
        raise ValueError(f"{label}.status is not consumable: {status!r}")
    _validate_real_date(value["last_valid_review_as_of"],
                        f"{label}.last_valid_review_as_of")
    watch = _validate_code_list(value["watch"], f"{label}.watch")
    eligible = _validate_code_list(value["exit_eligible"], f"{label}.exit_eligible")
    if set(watch) & set(eligible):
        raise ValueError(f"{label}.watch and {label}.exit_eligible must be disjoint")
    if (
        status in {"AVAILABLE", "REVIEW_BLOCKED_DATA"}
        and (watch or eligible)
        and value["last_valid_review_as_of"] is None
    ):
        raise ValueError(
            f"{label}.last_valid_review_as_of must be a real date when members exist"
        )

    error = value["error"]
    if status in {"AVAILABLE", "NOT_INITIALIZED"} and error is not None:
        raise ValueError(f"{label}.error must be None for status {status}")
    if status == "NOT_INITIALIZED" and (
            value["last_valid_review_as_of"] is not None or watch or eligible):
        raise ValueError(f"{label} NOT_INITIALIZED must not contain review state")
    if status == "REVIEW_BLOCKED_DATA" and (
            not isinstance(error, str)
            or not error.startswith(_BLOCKED_ERROR_PREFIX)
            or not error[len(_BLOCKED_ERROR_PREFIX):]):
        raise ValueError(
            f"{label}.error must use the stable REVIEW_BLOCKED_DATA prefix"
        )
    return value


def validate_decision_snapshot(
    snapshot: Any, *, source: str = "decision snapshot",
) -> dict[str, Any]:
    """Validate the minimal decision-row contract shared by all consumers."""
    if not isinstance(snapshot, dict):
        raise ValueError(f"{source} must be an object")
    decisions = snapshot.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError(f"{source} decisions must be a list")

    seen: set[str] = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"{source} decision[{index}] must be an object")
        code = decision.get("ts_code")
        if not isinstance(code, str) or not _TS_CODE.fullmatch(code):
            raise ValueError(f"{source} decision[{index}] has invalid ts_code")
        if code in seen:
            raise ValueError(f"{source} has duplicate decision ts_code {code}")
        seen.add(code)
        state = decision.get("state")
        if not isinstance(state, str) or state not in _DECISION_STATES:
            raise ValueError(
                f"{source} decision[{index}] has invalid state {state!r}"
            )
    return snapshot


def require_decision_snapshot_ready(
    snapshot: Any, *, source: str = "decision snapshot",
) -> dict[str, Any]:
    """Require a complete snapshot with a consumable C2 projection."""
    validate_decision_snapshot(snapshot, source=source)
    if snapshot.get("data_status") != "complete":
        raise ValueError(f"{source} is not complete")
    c2_state = snapshot.get("c2_state")
    if isinstance(c2_state, dict) and c2_state.get("status") == "UNAVAILABLE":
        raise ValueError(f"{source} is not complete: c2_state.status is UNAVAILABLE")
    validate_c2_projection(c2_state)
    return snapshot
