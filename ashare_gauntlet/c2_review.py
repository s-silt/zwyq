"""Pure state transitions for the production C2 monthly holding review."""
from __future__ import annotations

import copy
from datetime import datetime
import re
from typing import Any


SCHEMA = "c2_review_state.v1"
CONFIRMATIONS_REQUIRED = 2
OBSERVATION_STATUSES = frozenset({"INSIDE", "OUTSIDE", "BYPASS"})

_DATE_RE = re.compile(r"^\d{8}$")
_PERIOD_RE = re.compile(r"^\d{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POSITION_FIELDS = {
    "name", "out_streak", "status", "first_out_as_of",
    "last_valid_review_as_of", "exit_eligible_as_of",
}
_STATE_FIELDS = {
    "schema", "updated_at", "last_valid_review_period",
    "last_valid_review_as_of", "positions", "reviews",
}
_VALID_REVIEW_FIELDS = {
    "status", "period", "as_of", "decision_snapshot", "factor_snapshot",
    "observation_count", "transitions",
}
_BLOCKED_REVIEW_FIELDS = {
    "status", "period", "as_of", "issues", "evidence_hashes",
}
_TRANSITION_FIELDS = {
    "ts_code", "name", "from_status", "to_status", "action",
}
_POSITION_STATUSES = frozenset({"WATCH", "EXIT_ELIGIBLE"})
_TRANSITION_STATUSES = frozenset({"CLEAR", "WATCH", "EXIT_ELIGIBLE", "CLEARED", "REMOVED"})
_TRANSITION_ACTIONS = frozenset({
    "OUTSIDE_STARTED", "OUTSIDE_CONFIRMED", "OUTSIDE_CAPPED",
    "REENTERED_D10", "BYPASS", "HOLDING_REMOVED",
})
_REVIEW_STATUSES = frozenset({"VALID", "REVIEW_BLOCKED_DATA"})


class C2ReviewError(ValueError):
    """The C2 state or review evidence violates the persisted contract."""


def initial_state() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "updated_at": None,
        "last_valid_review_period": None,
        "last_valid_review_as_of": None,
        "positions": {},
        "reviews": [],
    }


def _require_exact_fields(value: dict, fields: set[str], label: str) -> None:
    actual = set(value)
    if actual != fields:
        raise C2ReviewError(
            f"{label} fields mismatch: missing={sorted(fields - actual, key=repr)}, "
            f"extra={sorted(actual - fields, key=repr)}"
        )


def _validate_string_enum(value: Any, choices: frozenset[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise C2ReviewError(f"{label} is invalid")
    return value


def _validate_period(value: Any, label: str = "period") -> str:
    if not isinstance(value, str) or not _PERIOD_RE.fullmatch(value):
        raise C2ReviewError(f"{label} must be a real YYYYMM period")
    try:
        datetime.strptime(value, "%Y%m")
    except ValueError as exc:
        raise C2ReviewError(f"{label} must be a real YYYYMM period") from exc
    return value


def _validate_date(value: Any, label: str = "as_of") -> str:
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise C2ReviewError(f"{label} must be a real YYYYMMDD date")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise C2ReviewError(f"{label} must be a real YYYYMMDD date") from exc
    return value


def _validate_period_date(period: Any, as_of: Any, *, prefix: str = "") -> tuple[str, str]:
    period_label = f"{prefix}period" if prefix else "period"
    as_of_label = f"{prefix}as_of" if prefix else "as_of"
    checked_period = _validate_period(period, period_label)
    checked_as_of = _validate_date(as_of, as_of_label)
    if checked_as_of[:6] != checked_period:
        raise C2ReviewError(f"{period_label} must be matching {as_of_label}")
    return checked_period, checked_as_of


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise C2ReviewError(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
    return value


def _validate_snapshot(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise C2ReviewError(f"{label} must be an object")
    _require_exact_fields(value, {"path", "sha256"}, label)
    if not isinstance(value["path"], str) or not value["path"]:
        raise C2ReviewError(f"{label}.path must be a non-empty string")
    _validate_sha256(value["sha256"], label)
    return value


def _validate_transition(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise C2ReviewError(f"{label} must be an object")
    _require_exact_fields(value, _TRANSITION_FIELDS, label)
    for field in ("ts_code", "name"):
        if not isinstance(value[field], str) or not value[field]:
            raise C2ReviewError(f"{label}.{field} must be a non-empty string")
    from_status = _validate_string_enum(
        value["from_status"], _TRANSITION_STATUSES, f"{label}.from_status"
    )
    to_status = _validate_string_enum(
        value["to_status"], _TRANSITION_STATUSES, f"{label}.to_status"
    )
    action = _validate_string_enum(value["action"], _TRANSITION_ACTIONS, f"{label}.action")
    exact_pairs = {
        "OUTSIDE_STARTED": ("CLEAR", "WATCH"),
        "OUTSIDE_CONFIRMED": ("WATCH", "EXIT_ELIGIBLE"),
        "OUTSIDE_CAPPED": ("EXIT_ELIGIBLE", "EXIT_ELIGIBLE"),
    }
    expected = exact_pairs.get(action)
    if expected is not None and (from_status, to_status) != expected:
        raise C2ReviewError(f"{label} action and statuses are inconsistent")
    if action in {"REENTERED_D10", "BYPASS"} and (
            from_status not in _POSITION_STATUSES or to_status != "CLEARED"):
        raise C2ReviewError(f"{label} action and statuses are inconsistent")
    if action == "HOLDING_REMOVED" and (
            from_status not in _POSITION_STATUSES or to_status != "REMOVED"):
        raise C2ReviewError(f"{label} action and statuses are inconsistent")


def _validate_hashes(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise C2ReviewError(f"{label} must be an object")
    for key, digest in value.items():
        if not isinstance(key, str) or not key:
            raise C2ReviewError(f"{label} keys must be non-empty strings")
        _validate_sha256(digest, f"{label}.{key}")


def _validate_review(value: Any, index: int) -> str:
    label = f"reviews[{index}]"
    if not isinstance(value, dict):
        raise C2ReviewError(f"{label} must be an object")
    status = _validate_string_enum(value.get("status"), _REVIEW_STATUSES, f"{label}.status")
    if status == "VALID":
        _require_exact_fields(value, _VALID_REVIEW_FIELDS, label)
        _validate_period_date(value["period"], value["as_of"], prefix=f"{label}.")
        _validate_snapshot(value["decision_snapshot"], f"{label}.decision_snapshot")
        _validate_snapshot(value["factor_snapshot"], f"{label}.factor_snapshot")
        count = value["observation_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise C2ReviewError(f"{label}.observation_count must be a non-negative integer")
        transitions = value["transitions"]
        if not isinstance(transitions, list):
            raise C2ReviewError(f"{label}.transitions must be a list")
        seen: set[str] = set()
        for transition_index, transition in enumerate(transitions):
            _validate_transition(transition, f"{label}.transitions[{transition_index}]")
            code = transition["ts_code"]
            if code in seen:
                raise C2ReviewError(f"{label}.transitions has duplicate ts_code {code}")
            seen.add(code)
        return status
    if status == "REVIEW_BLOCKED_DATA":
        _require_exact_fields(value, _BLOCKED_REVIEW_FIELDS, label)
        _validate_period_date(value["period"], value["as_of"], prefix=f"{label}.")
        issues = value["issues"]
        if (not isinstance(issues, list) or not issues
                or not all(isinstance(issue, str) and issue for issue in issues)):
            raise C2ReviewError(f"{label}.issues must be a non-empty string list")
        if len(set(issues)) != len(issues):
            raise C2ReviewError(f"{label}.issues must be unique")
        _validate_hashes(value["evidence_hashes"], f"{label}.evidence_hashes")
        return status
    raise AssertionError("unreachable review status")


def _validate_position(code: Any, value: Any, last_as_of: str | None,
                       valid_as_ofs: set[str]) -> None:
    label = f"positions[{code!r}]"
    if not isinstance(code, str) or not code:
        raise C2ReviewError("position codes must be non-empty strings")
    if not isinstance(value, dict):
        raise C2ReviewError(f"{label} must be an object")
    _require_exact_fields(value, _POSITION_FIELDS, label)
    if not isinstance(value["name"], str) or not value["name"]:
        raise C2ReviewError(f"{label}.name must be a non-empty string")
    streak = value["out_streak"]
    if not isinstance(streak, int) or isinstance(streak, bool) or streak not in {1, 2}:
        raise C2ReviewError(f"{label}.out_streak must be 1 or 2")
    status = _validate_string_enum(value["status"], _POSITION_STATUSES, f"{label}.status")
    first_out = _validate_date(value["first_out_as_of"], f"{label}.first_out_as_of")
    observed = _validate_date(
        value["last_valid_review_as_of"], f"{label}.last_valid_review_as_of"
    )
    if last_as_of is None or observed != last_as_of:
        raise C2ReviewError(f"{label}.last_valid_review_as_of must match state")
    if first_out > observed:
        raise C2ReviewError(f"{label}.first_out_as_of cannot follow last observation")
    if first_out not in valid_as_ofs:
        raise C2ReviewError(f"{label}.first_out_as_of must reference a valid review")
    eligible_as_of = value["exit_eligible_as_of"]
    if status == "WATCH":
        if streak != 1 or eligible_as_of is not None:
            raise C2ReviewError(f"{label} WATCH fields are inconsistent")
    else:
        if streak != CONFIRMATIONS_REQUIRED:
            raise C2ReviewError(f"{label} EXIT_ELIGIBLE streak is inconsistent")
        eligible = _validate_date(eligible_as_of, f"{label}.exit_eligible_as_of")
        if not first_out <= eligible <= observed:
            raise C2ReviewError(f"{label}.exit_eligible_as_of is inconsistent")
        if eligible not in valid_as_ofs:
            raise C2ReviewError(f"{label}.exit_eligible_as_of must reference a valid review")


def _replay_valid_reviews(valid_reviews: list[dict]) -> dict[str, dict[str, Any]]:
    active: dict[str, dict[str, Any]] = {}
    for review in valid_reviews:
        prior_codes = set(active)
        transitioned: set[str] = set()
        observation_driven = 0
        for transition in review["transitions"]:
            code = transition["ts_code"]
            name = transition["name"]
            action = transition["action"]
            prior = active.get(code)
            if action != "HOLDING_REMOVED":
                observation_driven += 1

            if action == "OUTSIDE_STARTED":
                if prior is not None:
                    raise C2ReviewError(
                        f"review {review['period']} OUTSIDE_STARTED requires no active position {code}"
                    )
                active[code] = {
                    "name": name,
                    "out_streak": 1,
                    "status": "WATCH",
                    "first_out_as_of": review["as_of"],
                    "last_valid_review_as_of": review["as_of"],
                    "exit_eligible_as_of": None,
                }
            elif action == "OUTSIDE_CONFIRMED":
                if prior is None or prior["status"] != "WATCH":
                    raise C2ReviewError(
                        f"review {review['period']} OUTSIDE_CONFIRMED requires WATCH {code}"
                    )
                active[code] = {
                    **prior,
                    "name": name,
                    "out_streak": CONFIRMATIONS_REQUIRED,
                    "status": "EXIT_ELIGIBLE",
                    "last_valid_review_as_of": review["as_of"],
                    "exit_eligible_as_of": review["as_of"],
                }
            elif action == "OUTSIDE_CAPPED":
                if prior is None or prior["status"] != "EXIT_ELIGIBLE":
                    raise C2ReviewError(
                        f"review {review['period']} OUTSIDE_CAPPED requires EXIT_ELIGIBLE {code}"
                    )
                active[code] = {
                    **prior,
                    "name": name,
                    "last_valid_review_as_of": review["as_of"],
                }
            else:
                if prior is None or transition["from_status"] != prior["status"]:
                    raise C2ReviewError(
                        f"review {review['period']} {action} prior state mismatch for {code}"
                    )
                del active[code]
            transitioned.add(code)

        missing = prior_codes - transitioned
        if missing:
            raise C2ReviewError(
                f"review {review['period']} does not account for active positions "
                f"{sorted(missing, key=repr)}"
            )
        if review["observation_count"] < observation_driven:
            raise C2ReviewError(
                f"review {review['period']} observation_count is smaller than "
                "observation-driven transitions"
            )
    return active


def validate_state(state: dict) -> None:
    """Validate the complete persisted state contract and its cross-field invariants."""
    if not isinstance(state, dict):
        raise C2ReviewError("state must be an object")
    _require_exact_fields(state, _STATE_FIELDS, "state")
    if state["schema"] != SCHEMA:
        raise C2ReviewError(f"state.schema must be {SCHEMA}")
    updated_at = state["updated_at"]
    if updated_at is not None:
        try:
            parsed_updated_at = (
                datetime.fromisoformat(updated_at) if isinstance(updated_at, str) else None
            )
        except ValueError:
            parsed_updated_at = None
        if parsed_updated_at is None or parsed_updated_at.utcoffset() is None:
            raise C2ReviewError(
                "state.updated_at must be null or a timezone-aware ISO-8601 timestamp"
            )

    period = state["last_valid_review_period"]
    as_of = state["last_valid_review_as_of"]
    if (period is None) != (as_of is None):
        raise C2ReviewError("last-valid period and as_of must both be null or both be set")
    if period is not None:
        _validate_period_date(period, as_of, prefix="state.last_valid_review_")

    if not isinstance(state["reviews"], list):
        raise C2ReviewError("state.reviews must be a list")
    valid_reviews: list[dict] = []
    for index, review in enumerate(state["reviews"]):
        if _validate_review(review, index) == "VALID":
            valid_reviews.append(review)
    valid_periods = [review["period"] for review in valid_reviews]
    if any(left >= right for left, right in zip(valid_periods, valid_periods[1:])):
        raise C2ReviewError("valid review ordering must be strictly increasing")
    if valid_reviews:
        last_review = valid_reviews[-1]
        if period != last_review["period"] or as_of != last_review["as_of"]:
            raise C2ReviewError("last-valid fields must match the latest valid review")
    elif period is not None:
        raise C2ReviewError("last-valid fields require a persisted valid review")

    if not isinstance(state["positions"], dict):
        raise C2ReviewError("state.positions must be an object")
    if state["positions"] and as_of is None:
        raise C2ReviewError("positions require a valid review")
    for code, position in state["positions"].items():
        _validate_position(code, position, as_of, {review["as_of"] for review in valid_reviews})
    reconstructed = _replay_valid_reviews(valid_reviews)
    if reconstructed != state["positions"]:
        raise C2ReviewError("state.positions does not match replayed valid review history")


def _validate_evidence(evidence: Any) -> None:
    if not isinstance(evidence, dict):
        raise C2ReviewError("evidence must be an object")
    _require_exact_fields(
        evidence,
        {"period", "as_of", "decision_snapshot", "factor_snapshot", "observations"},
        "evidence",
    )
    _validate_period_date(evidence["period"], evidence["as_of"])
    _validate_snapshot(evidence["decision_snapshot"], "decision_snapshot")
    _validate_snapshot(evidence["factor_snapshot"], "factor_snapshot")
    observations = evidence["observations"]
    if not isinstance(observations, list):
        raise C2ReviewError("observations must be a list")
    seen: set[str] = set()
    for index, observation in enumerate(observations):
        label = f"observations[{index}]"
        if not isinstance(observation, dict):
            raise C2ReviewError(f"{label} must be an object")
        _require_exact_fields(observation, {"ts_code", "name", "status"}, label)
        code = observation["ts_code"]
        if not isinstance(code, str) or not code:
            raise C2ReviewError(f"{label}.ts_code must be a non-empty string")
        if code in seen:
            raise C2ReviewError(f"observations has duplicate ts_code {code}")
        seen.add(code)
        if not isinstance(observation["name"], str) or not observation["name"]:
            raise C2ReviewError(f"{label}.name must be a non-empty string")
        _validate_string_enum(
            observation["status"], OBSERVATION_STATUSES, f"{label}.status"
        )


def _events_from_transitions(transitions: list[dict]) -> dict[str, list[str]]:
    events = {
        "newly_exit_eligible": [],
        "cleared_reentered": [],
        "cleared_bypass": [],
        "removed_holdings": [],
    }
    event_by_action = {
        "OUTSIDE_CONFIRMED": "newly_exit_eligible",
        "REENTERED_D10": "cleared_reentered",
        "BYPASS": "cleared_bypass",
        "HOLDING_REMOVED": "removed_holdings",
    }
    for transition in transitions:
        event = event_by_action.get(transition["action"])
        if event is not None:
            events[event].append(transition["ts_code"])
    for codes in events.values():
        codes.sort()
    return events


def _same_valid_review(review: dict, evidence: dict) -> bool:
    return (
        review["decision_snapshot"]["sha256"] == evidence["decision_snapshot"]["sha256"]
        and review["factor_snapshot"]["sha256"] == evidence["factor_snapshot"]["sha256"]
    )


def advance_review(state: dict, evidence: dict) -> tuple[dict, dict]:
    """Validate, copy, and advance one valid monthly observation."""
    validate_state(state)
    _validate_evidence(evidence)

    same_period = [
        review for review in state["reviews"]
        if review["status"] == "VALID" and review["period"] == evidence["period"]
    ]
    if same_period:
        review = same_period[0]
        if not _same_valid_review(review, evidence):
            raise C2ReviewError(f"valid review conflict for period {evidence['period']}")
        return copy.deepcopy(state), _events_from_transitions(review["transitions"])

    last_period = state["last_valid_review_period"]
    if last_period is not None and evidence["period"] <= last_period:
        raise C2ReviewError("valid review ordering must be strictly increasing")

    advanced = copy.deepcopy(state)
    prior_positions = copy.deepcopy(state["positions"])
    observed_codes = {observation["ts_code"] for observation in evidence["observations"]}
    transitions: list[dict[str, str]] = []

    for observation in sorted(evidence["observations"], key=lambda row: row["ts_code"]):
        code = observation["ts_code"]
        name = observation["name"]
        observation_status = observation["status"]
        prior = prior_positions.get(code)
        if observation_status == "OUTSIDE":
            prior_status = prior["status"] if prior else "CLEAR"
            streak = min((prior["out_streak"] if prior else 0) + 1, CONFIRMATIONS_REQUIRED)
            next_status = "EXIT_ELIGIBLE" if streak == CONFIRMATIONS_REQUIRED else "WATCH"
            first_out = prior["first_out_as_of"] if prior else evidence["as_of"]
            eligible_as_of = prior["exit_eligible_as_of"] if prior else None
            if next_status == "EXIT_ELIGIBLE" and eligible_as_of is None:
                eligible_as_of = evidence["as_of"]
            advanced["positions"][code] = {
                "name": name,
                "out_streak": streak,
                "status": next_status,
                "first_out_as_of": first_out,
                "last_valid_review_as_of": evidence["as_of"],
                "exit_eligible_as_of": eligible_as_of,
            }
            if prior_status == "CLEAR":
                action = "OUTSIDE_STARTED"
            elif prior_status == "WATCH":
                action = "OUTSIDE_CONFIRMED"
            else:
                action = "OUTSIDE_CAPPED"
            transitions.append({
                "ts_code": code,
                "name": name,
                "from_status": prior_status,
                "to_status": next_status,
                "action": action,
            })
        elif prior is not None:
            del advanced["positions"][code]
            transitions.append({
                "ts_code": code,
                "name": name,
                "from_status": prior["status"],
                "to_status": "CLEARED",
                "action": "REENTERED_D10" if observation_status == "INSIDE" else "BYPASS",
            })

    for code in sorted(set(prior_positions) - observed_codes):
        prior = prior_positions[code]
        del advanced["positions"][code]
        transitions.append({
            "ts_code": code,
            "name": prior["name"],
            "from_status": prior["status"],
            "to_status": "REMOVED",
            "action": "HOLDING_REMOVED",
        })

    review = {
        "status": "VALID",
        "period": evidence["period"],
        "as_of": evidence["as_of"],
        "decision_snapshot": copy.deepcopy(evidence["decision_snapshot"]),
        "factor_snapshot": copy.deepcopy(evidence["factor_snapshot"]),
        "observation_count": len(evidence["observations"]),
        "transitions": transitions,
    }
    advanced["reviews"].append(review)
    advanced["last_valid_review_period"] = evidence["period"]
    advanced["last_valid_review_as_of"] = evidence["as_of"]
    events = _events_from_transitions(transitions)
    validate_state(advanced)
    return advanced, events


def record_blocked_review(state: dict, *, period: str, as_of: str,
                          issues: list[str], evidence_hashes: dict[str, str]) -> dict:
    """Append one deduplicated blocked audit without changing active C2 state."""
    validate_state(state)
    _validate_period_date(period, as_of)
    if (not isinstance(issues, list) or not issues
            or not all(isinstance(issue, str) and issue for issue in issues)):
        raise C2ReviewError("issues must be a non-empty string list")
    if len(set(issues)) != len(issues):
        raise C2ReviewError("issues must be unique")
    _validate_hashes(evidence_hashes, "evidence_hashes")

    blocked = {
        "status": "REVIEW_BLOCKED_DATA",
        "period": period,
        "as_of": as_of,
        "issues": copy.deepcopy(issues),
        "evidence_hashes": copy.deepcopy(evidence_hashes),
    }
    for review in state["reviews"]:
        if (review["status"] == "REVIEW_BLOCKED_DATA"
                and review["period"] == period
                and sorted(review["issues"]) == sorted(issues)
                and review["evidence_hashes"] == evidence_hashes):
            return copy.deepcopy(state)
    advanced = copy.deepcopy(state)
    advanced["reviews"].append(blocked)
    validate_state(advanced)
    return advanced


def eligible_codes(state: dict) -> set[str]:
    validate_state(state)
    return {
        code for code, row in state["positions"].items()
        if row["status"] == "EXIT_ELIGIBLE"
    }
