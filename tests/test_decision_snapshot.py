from __future__ import annotations

import copy

import pytest

from ashare_gauntlet.decision_snapshot import require_decision_snapshot_ready


def snapshot() -> dict:
    return {
        "data_status": "complete",
        "c2_state": {
            "status": "NOT_INITIALIZED",
            "last_valid_review_as_of": None,
            "watch": [],
            "exit_eligible": [],
            "error": None,
        },
        "decisions": [],
    }


@pytest.mark.parametrize(("decisions", "message"), [
    (None, "must be a list"),
    ([None], "must be an object"),
    ([{"ts_code": "A", "state": "WAIT"}], "invalid ts_code"),
    ([{"ts_code": "600001.sh", "state": "WAIT"}], "invalid ts_code"),
    ([{"ts_code": "600001.SH", "state": "UNKNOWN"}], "invalid state"),
    ([{"ts_code": "600001.SH", "state": []}], "invalid state"),
    ([
        {"ts_code": "600001.SH", "state": "WAIT"},
        {"ts_code": "600001.SH", "state": "HOLD"},
    ], "duplicate"),
])
def test_ready_snapshot_rejects_invalid_minimal_decision_contract(
        decisions: object, message: str) -> None:
    value = snapshot()
    value["decisions"] = decisions

    with pytest.raises(ValueError, match=message):
        require_decision_snapshot_ready(value)


def test_ready_snapshot_accepts_all_decision_states() -> None:
    value = snapshot()
    value["decisions"] = [
        {"ts_code": f"60000{index}.SH", "state": state}
        for index, state in enumerate(("BUY", "WAIT", "HOLD", "EXIT"), start=1)
    ]

    assert require_decision_snapshot_ready(copy.deepcopy(value)) == value
