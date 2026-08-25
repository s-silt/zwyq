from __future__ import annotations

import copy

import pytest

from ashare_gauntlet.c2_review import (
    C2ReviewError,
    advance_review,
    eligible_codes,
    initial_state,
    record_blocked_review,
)


def evidence(period: str, as_of: str, observations: list[dict]) -> dict:
    return {
        "period": period,
        "as_of": as_of,
        "decision_snapshot": {"path": f"{as_of}_buy_decisions.json", "sha256": "d" * 64},
        "factor_snapshot": {"path": f"{as_of}_factor.json", "sha256": "f" * 64},
        "observations": observations,
    }


def outside(code: str = "A", name: str = "甲") -> dict:
    return {"ts_code": code, "name": name, "status": "OUTSIDE"}


def test_two_valid_outside_reviews_make_exit_eligible() -> None:
    first, first_events = advance_review(initial_state(), evidence(
        "202601", "20260130", [{"ts_code": "A", "name": "甲", "status": "OUTSIDE"}]))
    second, second_events = advance_review(first, evidence(
        "202602", "20260227", [{"ts_code": "A", "name": "甲", "status": "OUTSIDE"}]))
    assert first["positions"]["A"]["out_streak"] == 1
    assert first["positions"]["A"]["status"] == "WATCH"
    assert second["positions"]["A"]["out_streak"] == 2
    assert second["positions"]["A"]["status"] == "EXIT_ELIGIBLE"
    assert eligible_codes(second) == {"A"}
    assert second_events["newly_exit_eligible"] == ["A"]
    assert first_events["newly_exit_eligible"] == []


def test_reentry_clears_and_blocked_month_preserves_streak() -> None:
    state, _ = advance_review(initial_state(), evidence(
        "202601", "20260130", [{"ts_code": "A", "name": "甲", "status": "OUTSIDE"}]))
    blocked = record_blocked_review(state, period="202602", as_of="20260227",
                                    issues=["CORE_EOD_MISSING"], evidence_hashes={})
    assert blocked["positions"]["A"]["out_streak"] == 1
    cleared, events = advance_review(blocked, evidence(
        "202603", "20260331", [{"ts_code": "A", "name": "甲", "status": "INSIDE"}]))
    assert "A" not in cleared["positions"]
    assert events["cleared_reentered"] == ["A"]
    assert blocked["last_valid_review_period"] == "202601"


def test_bypass_clears_active_position() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    cleared, events = advance_review(
        state,
        evidence("202602", "20260227", [{"ts_code": "A", "name": "甲", "status": "BYPASS"}]),
    )
    assert "A" not in cleared["positions"]
    assert events["cleared_bypass"] == ["A"]


def test_removed_holding_is_archived_by_transition() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    advanced, events = advance_review(state, evidence(
        "202602", "20260227", [{"ts_code": "B", "name": "乙", "status": "INSIDE"}]))
    assert "A" not in advanced["positions"]
    assert events["removed_holdings"] == ["A"]
    assert advanced["reviews"][-1]["transitions"] == [{
        "ts_code": "A", "name": "甲", "from_status": "WATCH", "to_status": "REMOVED",
        "action": "HOLDING_REMOVED",
    }]


def test_replay_rejects_removed_holding_name_mismatch() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    state, _ = advance_review(state, evidence("202602", "20260227", []))
    corrupt = copy.deepcopy(state)
    corrupt["reviews"][-1]["transitions"][0]["name"] = "乙"
    with pytest.raises(C2ReviewError):
        eligible_codes(corrupt)


def test_skipped_month_can_later_advance_streak() -> None:
    first, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    third, events = advance_review(first, evidence("202603", "20260331", [outside()]))
    assert third["positions"]["A"]["out_streak"] == 2
    assert events["newly_exit_eligible"] == ["A"]


def test_identical_valid_review_replay_is_idempotent() -> None:
    review = evidence("202601", "20260130", [outside()])
    first, first_events = advance_review(initial_state(), review)
    replayed, replay_events = advance_review(first, copy.deepcopy(review))
    assert replayed == first
    assert replayed is not first
    assert replay_events == first_events
    assert replayed["reviews"][-1]["status"] == "VALID"
    assert replayed["reviews"][-1]["decision_snapshot"] == review["decision_snapshot"]


def test_valid_replay_identity_uses_period_and_source_hashes() -> None:
    review = evidence("202601", "20260130", [outside()])
    state, expected_events = advance_review(initial_state(), review)
    replay = copy.deepcopy(review)
    replay["decision_snapshot"]["path"] = "relocated_decisions.json"
    replay["factor_snapshot"]["path"] = "relocated_factor.json"
    replayed, events = advance_review(state, replay)
    assert replayed == state
    assert events == expected_events


def test_same_period_changed_hash_conflicts() -> None:
    review = evidence("202601", "20260130", [outside()])
    state, _ = advance_review(initial_state(), review)
    changed = copy.deepcopy(review)
    changed["decision_snapshot"]["sha256"] = "a" * 64
    with pytest.raises(C2ReviewError, match="conflict"):
        advance_review(state, changed)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda row: row["decision_snapshot"].update(sha256="A" * 64), "sha256"),
        (lambda row: row["factor_snapshot"].update(sha256="f" * 63), "sha256"),
        (lambda row: row.update(as_of="20260230"), "as_of"),
        (lambda row: row.update(period="202613"), "period"),
        (lambda row: row.update(period="202602"), "matching"),
        (lambda row: row["observations"][0].update(status="UNKNOWN"), "status"),
    ],
)
def test_malformed_evidence_is_rejected(mutate, match: str) -> None:
    row = evidence("202601", "20260130", [outside()])
    mutate(row)
    with pytest.raises(C2ReviewError, match=match):
        advance_review(initial_state(), row)


def test_duplicate_observation_codes_are_rejected() -> None:
    with pytest.raises(C2ReviewError, match="duplicate"):
        advance_review(initial_state(), evidence(
            "202601", "20260130", [outside(), outside(name="重复")]))


@pytest.mark.parametrize(
    "corrupt",
    [
        {"schema": "wrong", "updated_at": None, "last_valid_review_period": None,
         "last_valid_review_as_of": None, "positions": {}, "reviews": []},
        {**initial_state(), "positions": {"A": {"name": "甲", "out_streak": 3,
                                                   "status": "EXIT_ELIGIBLE"}}},
        {**initial_state(), "reviews": [{"kind": "BLOCKED"}]},
        {**initial_state(), "updated_at": 123},
    ],
)
def test_corrupt_state_is_rejected(corrupt: dict) -> None:
    with pytest.raises(C2ReviewError):
        eligible_codes(corrupt)
    with pytest.raises(C2ReviewError):
        advance_review(corrupt, evidence("202601", "20260130", []))


def test_semantically_corrupt_persisted_fields_are_rejected() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    bad_transition = copy.deepcopy(state)
    bad_transition["reviews"][0]["transitions"][0].update(
        from_status="EXIT_ELIGIBLE", to_status="REMOVED", action="OUTSIDE_STARTED"
    )
    with pytest.raises(C2ReviewError):
        eligible_codes(bad_transition)

    bad_position_date = copy.deepcopy(state)
    bad_position_date["positions"]["A"]["first_out_as_of"] = "20260129"
    with pytest.raises(C2ReviewError):
        eligible_codes(bad_position_date)


def test_replay_rejects_exit_eligible_without_confirming_transition() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    state, _ = advance_review(state, evidence("202602", "20260227", [outside()]))
    corrupt = copy.deepcopy(state)
    corrupt["reviews"][-1]["transitions"] = []
    with pytest.raises(C2ReviewError):
        eligible_codes(corrupt)


def test_replay_rejects_eligibility_on_first_outside_review() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    state, _ = advance_review(state, evidence("202602", "20260227", [outside()]))
    corrupt = copy.deepcopy(state)
    corrupt["positions"]["A"]["exit_eligible_as_of"] = "20260130"
    with pytest.raises(C2ReviewError):
        eligible_codes(corrupt)


def test_replay_rejects_transition_exceeding_observation_count() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    corrupt = copy.deepcopy(state)
    corrupt["reviews"][0]["observation_count"] = 0
    with pytest.raises(C2ReviewError):
        eligible_codes(corrupt)


def test_replay_rejects_watch_with_unrelated_first_out_review() -> None:
    inside = {"ts_code": "A", "name": "甲", "status": "INSIDE"}
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [inside]))
    state, _ = advance_review(state, evidence("202602", "20260227", [outside()]))
    corrupt = copy.deepcopy(state)
    corrupt["positions"]["A"]["first_out_as_of"] = "20260130"
    with pytest.raises(C2ReviewError):
        eligible_codes(corrupt)


def test_replay_rejects_position_name_disagreeing_with_latest_transition() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    corrupt = copy.deepcopy(state)
    corrupt["positions"]["A"]["name"] = "乙"
    with pytest.raises(C2ReviewError):
        eligible_codes(corrupt)


def test_replay_rejects_clear_transition_with_wrong_prior_status() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    inside = {"ts_code": "A", "name": "甲", "status": "INSIDE"}
    state, _ = advance_review(state, evidence("202602", "20260227", [inside]))
    corrupt = copy.deepcopy(state)
    corrupt["reviews"][-1]["transitions"][0]["from_status"] = "EXIT_ELIGIBLE"
    with pytest.raises(C2ReviewError):
        eligible_codes(corrupt)


@pytest.mark.parametrize(
    ("target", "malformed"),
    [
        ("observation", []),
        ("position", {}),
        ("transition_from", []),
        ("transition_action", {}),
    ],
)
def test_unhashable_enum_values_raise_domain_error(target: str, malformed) -> None:
    if target == "observation":
        review = evidence("202601", "20260130", [outside()])
        review["observations"][0]["status"] = malformed
        operation = lambda: advance_review(initial_state(), review)
    else:
        state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
        if target == "position":
            state["positions"]["A"]["status"] = malformed
        elif target == "transition_from":
            state["reviews"][0]["transitions"][0]["from_status"] = malformed
        else:
            state["reviews"][0]["transitions"][0]["action"] = malformed
        operation = lambda: eligible_codes(state)
    with pytest.raises(C2ReviewError):
        operation()


@pytest.mark.parametrize("target", ["state", "evidence"])
def test_mixed_unexpected_keys_raise_domain_error(target: str) -> None:
    if target == "state":
        malformed = initial_state()
        malformed[1] = "integer key"
        malformed["extra"] = "string key"
        operation = lambda: eligible_codes(malformed)
    else:
        malformed = evidence("202601", "20260130", [outside()])
        malformed[1] = "integer key"
        malformed["extra"] = "string key"
        operation = lambda: advance_review(initial_state(), malformed)
    with pytest.raises(C2ReviewError):
        operation()


def test_repeated_identical_blocked_attempt_is_deduplicated() -> None:
    state = record_blocked_review(
        initial_state(), period="202601", as_of="20260130",
        issues=["CORE_EOD_MISSING"], evidence_hashes={"daily": "a" * 64},
    )
    replayed = record_blocked_review(
        state, period="202601", as_of="20260130",
        issues=["CORE_EOD_MISSING"], evidence_hashes={"daily": "a" * 64},
    )
    assert replayed == state
    assert replayed is not state
    assert len(replayed["reviews"]) == 1
    assert replayed["reviews"][0]["status"] == "REVIEW_BLOCKED_DATA"


def test_blocked_deduplication_ignores_as_of_and_preserves_first_record() -> None:
    first = record_blocked_review(
        initial_state(), period="202601", as_of="20260115",
        issues=["CORE_EOD_MISSING"], evidence_hashes={"daily": "a" * 64},
    )
    replayed = record_blocked_review(
        first, period="202601", as_of="20260130",
        issues=["CORE_EOD_MISSING"], evidence_hashes={"daily": "a" * 64},
    )
    assert replayed == first
    assert len(replayed["reviews"]) == 1
    assert replayed["reviews"][0]["as_of"] == "20260115"


def test_validate_state_rejects_duplicate_normalized_blocked_identity() -> None:
    state = record_blocked_review(
        initial_state(), period="202601", as_of="20260115",
        issues=["FACTOR_MISSING", "CORE_EOD_MISSING"],
        evidence_hashes={"daily": "a" * 64},
    )
    duplicate = copy.deepcopy(state["reviews"][0])
    duplicate["as_of"] = "20260130"
    duplicate["issues"] = ["CORE_EOD_MISSING", "FACTOR_MISSING"]
    state["reviews"].append(duplicate)
    with pytest.raises(C2ReviewError):
        eligible_codes(state)


def test_repaired_valid_review_is_allowed_after_same_period_block() -> None:
    blocked = record_blocked_review(
        initial_state(), period="202601", as_of="20260130",
        issues=["CORE_EOD_MISSING"], evidence_hashes={},
    )
    repaired, _ = advance_review(blocked, evidence("202601", "20260130", [outside()]))
    assert [row["status"] for row in repaired["reviews"]] == [
        "REVIEW_BLOCKED_DATA", "VALID",
    ]
    assert repaired["last_valid_review_period"] == "202601"


def test_already_eligible_code_remains_capped_at_two() -> None:
    state, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    state, _ = advance_review(state, evidence("202602", "20260227", [outside()]))
    third, events = advance_review(state, evidence("202603", "20260331", [outside()]))
    assert third["positions"]["A"]["out_streak"] == 2
    assert third["positions"]["A"]["status"] == "EXIT_ELIGIBLE"
    assert events["newly_exit_eligible"] == []


def test_position_persists_full_audit_fields() -> None:
    first, _ = advance_review(initial_state(), evidence("202601", "20260130", [outside()]))
    assert first["positions"]["A"] == {
        "name": "甲",
        "out_streak": 1,
        "status": "WATCH",
        "first_out_as_of": "20260130",
        "last_valid_review_as_of": "20260130",
        "exit_eligible_as_of": None,
    }
    second, _ = advance_review(first, evidence("202602", "20260227", [outside()]))
    assert second["positions"]["A"]["first_out_as_of"] == "20260130"
    assert second["positions"]["A"]["last_valid_review_as_of"] == "20260227"
    assert second["positions"]["A"]["exit_eligible_as_of"] == "20260227"


def test_valid_review_rejects_non_increasing_period() -> None:
    state, _ = advance_review(initial_state(), evidence("202602", "20260227", [outside()]))
    with pytest.raises(C2ReviewError, match="ordering"):
        advance_review(state, evidence("202601", "20260130", [outside()]))


def test_updated_at_requires_aware_iso_timestamp_and_is_preserved() -> None:
    naive = {**initial_state(), "updated_at": "2026-01-30T16:00:00"}
    with pytest.raises(C2ReviewError, match="updated_at"):
        eligible_codes(naive)

    stamped = {**initial_state(), "updated_at": "2026-01-30T16:00:00+08:00"}
    advanced, _ = advance_review(stamped, evidence("202601", "20260130", [outside()]))
    assert advanced["updated_at"] == stamped["updated_at"]


def test_operations_never_mutate_inputs() -> None:
    state = initial_state()
    review = evidence("202601", "20260130", [outside()])
    before_state = copy.deepcopy(state)
    before_review = copy.deepcopy(review)
    advanced, _ = advance_review(state, review)
    assert state == before_state
    assert review == before_review

    before_advanced = copy.deepcopy(advanced)
    hashes = {"daily": "a" * 64}
    before_hashes = copy.deepcopy(hashes)
    record_blocked_review(advanced, period="202602", as_of="20260227",
                          issues=["CORE_EOD_MISSING"], evidence_hashes=hashes)
    assert advanced == before_advanced
    assert hashes == before_hashes
