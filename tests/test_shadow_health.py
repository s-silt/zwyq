from __future__ import annotations

import pytest

from ashare_gauntlet.shadow_health import ShadowHealthError, build_shadow_report


def test_three_negative_valid_months_require_review() -> None:
    rows = [
        {"date": "20260130", "ret_PROD": 0.01, "mkt_fwd": 0.02,
         "TO_PROD": 0.5, "cost_rt": 0.004},
        {"date": "20260227", "ret_PROD": 0.00, "mkt_fwd": 0.01,
         "TO_PROD": 0.5, "cost_rt": 0.004},
        {"date": "20260331", "ret_PROD": 0.01, "mkt_fwd": 0.02,
         "TO_PROD": 0.5, "cost_rt": 0.004},
    ]

    report = build_shadow_report(rows)

    assert report["schema"] == "production_shadow_health.v1"
    assert report["caliber_version"] == "prod_ep_bp_ivol_monthly_open_to_open.v1"
    assert report["observations"][-1]["transaction_cost"] == pytest.approx(0.002)
    assert report["observations"][-1]["net_excess_return"] == pytest.approx(-0.012)
    assert report["observations"][-1]["negative_valid_streak"] == 3
    assert report["current_negative_valid_streak"] == 3
    assert report["review_required"] is True
    assert report["production_policy_changed"] is False


def test_invalid_observation_breaks_negative_month_continuity() -> None:
    rows = [
        {"date": "20260130", "ret_PROD": 0.00, "mkt_fwd": 0.01,
         "TO_PROD": 0.5, "cost_rt": 0.004},
        {"date": "20260227", "ret_PROD": 0.00, "mkt_fwd": 0.01,
         "TO_PROD": 0.5},
        {"date": "20260331", "ret_PROD": 0.00, "mkt_fwd": 0.01,
         "TO_PROD": 0.5, "cost_rt": 0.004},
        {"date": "20260430", "ret_PROD": 0.00, "mkt_fwd": 0.01,
         "TO_PROD": 0.5, "cost_rt": 0.004},
    ]

    report = build_shadow_report(rows)

    assert [row["status"] for row in report["observations"]] == [
        "VALID", "INVALID", "VALID", "VALID"]
    assert report["observations"][1]["issues"] == ["cost_rt_missing_or_non_finite"]
    assert report["current_negative_valid_streak"] == 2
    assert report["review_required"] is False


def test_missing_calendar_month_is_explicit_and_breaks_continuity() -> None:
    rows = [
        {"date": "20260130", "ret_PROD": 0.00, "mkt_fwd": 0.01,
         "TO_PROD": 0.5, "cost_rt": 0.004},
        {"date": "20260331", "ret_PROD": 0.00, "mkt_fwd": 0.01,
         "TO_PROD": 0.5, "cost_rt": 0.004},
    ]

    report = build_shadow_report(rows)

    assert report["observation_count"] == 3
    gap = report["observations"][1]
    assert gap == {
        "period": "202602",
        "period_end": None,
        "status": "INVALID",
        "issues": ["missing_source_month"],
        "portfolio_return": None,
        "universe_equal_return": None,
        "turnover_tau": None,
        "round_trip_cost_rate": None,
        "transaction_cost": None,
        "gross_excess_return": None,
        "net_excess_return": None,
        "negative_valid_streak": 0,
        "review_required": False,
    }
    assert report["current_negative_valid_streak"] == 1


@pytest.mark.parametrize("bad_date", ["2026-01-30", "20260230", "", None])
def test_bad_period_end_fails_loud(bad_date: object) -> None:
    with pytest.raises(ShadowHealthError, match="YYYYMMDD"):
        build_shadow_report([{
            "date": bad_date, "ret_PROD": 0.01, "mkt_fwd": 0.0,
            "TO_PROD": 0.5, "cost_rt": 0.004,
        }])


@pytest.mark.parametrize("dates", [
    ("20260227", "20260130"),
    ("20260130", "20260130"),
    ("20260115", "20260130"),
])
def test_unsorted_duplicate_or_duplicate_month_fails_loud(dates: tuple[str, str]) -> None:
    rows = [
        {"date": date, "ret_PROD": 0.01, "mkt_fwd": 0.0,
         "TO_PROD": 0.5, "cost_rt": 0.004}
        for date in dates
    ]
    with pytest.raises(ShadowHealthError, match="strictly increasing"):
        build_shadow_report(rows)


def test_non_finite_and_negative_cost_inputs_are_invalid() -> None:
    rows = [
        {"date": "20260130", "ret_PROD": float("nan"), "mkt_fwd": 0.0,
         "TO_PROD": 0.5, "cost_rt": 0.004},
        {"date": "20260227", "ret_PROD": 0.01, "mkt_fwd": 0.0,
         "TO_PROD": -0.1, "cost_rt": -0.004},
    ]

    report = build_shadow_report(rows)

    assert report["invalid_count"] == 2
    assert report["observations"][0]["issues"] == ["ret_PROD_missing_or_non_finite"]
    assert report["observations"][1]["issues"] == ["TO_PROD_negative", "cost_rt_negative"]


def test_integer_too_large_for_float_is_invalid() -> None:
    report = build_shadow_report([{
        "date": "20260130", "ret_PROD": 10**10000, "mkt_fwd": 0.0,
        "TO_PROD": 0.5, "cost_rt": 0.004,
    }])

    observation = report["observations"][0]
    assert observation["status"] == "INVALID"
    assert observation["issues"] == ["ret_PROD_missing_or_non_finite"]
    assert report["current_negative_valid_streak"] == 0


def test_non_finite_derived_values_are_invalid_and_break_streak() -> None:
    rows = [
        {"date": "20260130", "ret_PROD": 0.0, "mkt_fwd": 0.01,
         "TO_PROD": 0.5, "cost_rt": 0.004},
        {"date": "20260227", "ret_PROD": 0.0, "mkt_fwd": 0.0,
         "TO_PROD": 1e308, "cost_rt": 1e308},
    ]

    report = build_shadow_report(rows)

    observation = report["observations"][1]
    assert observation["status"] == "INVALID"
    assert observation["issues"] == [
        "transaction_cost_non_finite", "net_excess_return_non_finite"]
    assert observation["transaction_cost"] is None
    assert observation["net_excess_return"] is None
    assert report["current_negative_valid_streak"] == 0
