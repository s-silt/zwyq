"""X-12 production-caliber monthly shadow return health (pure functions)."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Sequence


SCHEMA = "production_shadow_health.v1"
CALIBER_VERSION = "prod_ep_bp_ivol_monthly_open_to_open.v1"
NEGATIVE_MONTHS_FOR_REVIEW = 3
_SOURCE_FIELDS = ("ret_PROD", "mkt_fwd", "TO_PROD", "cost_rt")


class ShadowHealthError(ValueError):
    """The monthly source cannot be interpreted without guessing."""


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _period_end(value: Any) -> datetime:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ShadowHealthError(f"period end must be a real YYYYMMDD value, got {value!r}")
    try:
        return datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ShadowHealthError(
            f"period end must be a real YYYYMMDD value, got {value!r}") from exc


def _month_number(value: datetime) -> int:
    return value.year * 12 + value.month - 1


def _month_key(number: int) -> str:
    year, month0 = divmod(number, 12)
    return f"{year:04d}{month0 + 1:02d}"


def _invalid_gap(period: str) -> dict[str, Any]:
    return {
        "period": period,
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


def build_shadow_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build X-12 observations without changing production policy."""
    observations: list[dict[str, Any]] = []
    streak = 0
    previous_date: datetime | None = None
    previous_month: int | None = None
    for row in rows:
        if not isinstance(row, dict):
            raise ShadowHealthError(f"source row must be an object, got {type(row).__name__}")
        current_date = _period_end(row.get("date"))
        current_month = _month_number(current_date)
        if (previous_date is not None
                and (current_date <= previous_date or current_month == previous_month)):
            raise ShadowHealthError(
                "period ends must be strictly increasing with at most one row per month")
        if previous_month is not None:
            for missing in range(previous_month + 1, current_month):
                observations.append(_invalid_gap(_month_key(missing)))
                streak = 0

        values = {name: _finite(row.get(name)) for name in _SOURCE_FIELDS}
        issues = [f"{name}_missing_or_non_finite" for name, value in values.items()
                  if value is None]
        turnover = values["TO_PROD"]
        cost_rate = values["cost_rt"]
        if turnover is not None and turnover < 0:
            issues.append("TO_PROD_negative")
        if cost_rate is not None and cost_rate < 0:
            issues.append("cost_rt_negative")
        portfolio_return = values["ret_PROD"]
        universe_return = values["mkt_fwd"]
        transaction_cost = gross_excess = net_excess = None
        if not issues:
            transaction_cost = turnover * cost_rate
            gross_excess = portfolio_return - universe_return
            net_excess = gross_excess - transaction_cost
            for name, value in (
                    ("transaction_cost", transaction_cost),
                    ("gross_excess_return", gross_excess),
                    ("net_excess_return", net_excess)):
                if not math.isfinite(value):
                    issues.append(f"{name}_non_finite")
        valid = not issues
        if valid:
            streak = streak + 1 if net_excess < 0 else 0
        else:
            transaction_cost = gross_excess = net_excess = None
            streak = 0
        observations.append({
            "period": current_date.strftime("%Y%m"),
            "period_end": current_date.strftime("%Y%m%d"),
            "status": "VALID" if valid else "INVALID",
            "issues": issues,
            "portfolio_return": portfolio_return,
            "universe_equal_return": universe_return,
            "turnover_tau": turnover,
            "round_trip_cost_rate": cost_rate,
            "transaction_cost": transaction_cost,
            "gross_excess_return": gross_excess,
            "net_excess_return": net_excess,
            "negative_valid_streak": streak,
            "review_required": streak >= NEGATIVE_MONTHS_FOR_REVIEW,
        })
        previous_date = current_date
        previous_month = current_month
    return {
        "schema": SCHEMA,
        "caliber_version": CALIBER_VERSION,
        "production_policy_changed": False,
        "trigger": {
            "consecutive_negative_valid_months": NEGATIVE_MONTHS_FOR_REVIEW,
            "action": "human_review_only",
        },
        "observation_count": len(observations),
        "valid_count": sum(row["status"] == "VALID" for row in observations),
        "invalid_count": sum(row["status"] == "INVALID" for row in observations),
        "current_negative_valid_streak": streak,
        "review_required": streak >= NEGATIVE_MONTHS_FOR_REVIEW,
        "observations": observations,
    }
