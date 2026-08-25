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
_TS_CODE = re.compile(r"^\d{6}\.(?:SH|SZ)$")
_DATE8 = re.compile(r"^\d{8}$")
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
    if not all(isinstance(code, str) and _TS_CODE.fullmatch(code) for code in value):
        raise ValueError(f"{label} must contain valid ts_code strings")
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


def require_decision_snapshot_ready(
    snapshot: Any, *, source: str = "decision snapshot",
) -> dict[str, Any]:
    """Require a complete snapshot with a consumable C2 projection."""
    if not isinstance(snapshot, dict):
        raise ValueError(f"{source} must be an object")
    if snapshot.get("data_status") != "complete":
        raise ValueError(f"{source} is not complete")
    c2_state = snapshot.get("c2_state")
    if isinstance(c2_state, dict) and c2_state.get("status") == "UNAVAILABLE":
        raise ValueError(f"{source} is not complete: c2_state.status is UNAVAILABLE")
    validate_c2_projection(c2_state)
    return snapshot
