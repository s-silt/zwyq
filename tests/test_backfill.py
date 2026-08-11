from pathlib import Path

import pandas as pd
import pytest

from ashare_gauntlet.data.fetch import MARKET_ENDPOINTS, TokenExpiredError
from scripts import backfill


def _calendar(*days: str) -> pd.DataFrame:
    return pd.DataFrame({"cal_date": list(days), "is_open": [1] * len(days)})


def test_strict_backfill_covers_every_open_day_and_core_endpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        backfill,
        "fetch_trade_cal",
        lambda pro, start, end, cache_dir, strict=False: _calendar("20260806", "20260807"),
    )
    monkeypatch.setattr(
        backfill,
        "fetch_market_day",
        lambda pro, endpoint, day, cache_dir: calls.append((day, endpoint)),
    )

    result = backfill.run_backfill(
        object(), "20260806", "20260807", tmp_path, strict_market=True, max_workers=1,
    )

    assert result["ok"] is True
    assert result["calendar_status"] == "complete"
    assert result["required_endpoints"] == list(MARKET_ENDPOINTS)
    assert result["expected_pairs"] == result["completed_pairs"] == 8
    assert set(calls) == {
        (day, endpoint)
        for day in ("20260806", "20260807")
        for endpoint in MARKET_ENDPOINTS
    }


def test_strict_backfill_reports_endpoint_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        backfill,
        "fetch_trade_cal",
        lambda pro, start, end, cache_dir, strict=False: _calendar("20260807"),
    )

    def fetch(pro, endpoint, day, cache_dir):
        if endpoint == "daily_basic":
            raise RuntimeError("source down")

    monkeypatch.setattr(backfill, "fetch_market_day", fetch)
    result = backfill.run_backfill(
        object(), "20260807", "20260807", tmp_path, strict_market=True, max_workers=1,
    )

    assert result["ok"] is False
    assert result["completed_pairs"] == 3
    assert result["failed_pairs"] == [{
        "trade_date": "20260807",
        "endpoint": "daily_basic",
        "error_type": "RuntimeError",
        "error": "source down",
    }]


def test_strict_backfill_treats_token_expiry_as_fatal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        backfill,
        "fetch_trade_cal",
        lambda pro, start, end, cache_dir, strict=False: _calendar("20260807"),
    )
    monkeypatch.setattr(
        backfill,
        "fetch_market_day",
        lambda pro, endpoint, day, cache_dir: (_ for _ in ()).throw(TokenExpiredError("expired")),
    )

    result = backfill.run_backfill(
        object(), "20260807", "20260807", tmp_path, strict_market=True, max_workers=1,
    )

    assert result["ok"] is False
    assert result["fatal_error"]["error_type"] == "TokenExpiredError"


def test_strict_trade_calendar_failure_does_not_fallback(tmp_path: Path) -> None:
    class Pro:
        def trade_cal(self, **kwargs):
            raise ValueError("calendar down")

    try:
        backfill.fetch_trade_cal(Pro(), "20260806", "20260807", tmp_path, strict=True)
    except backfill.TradeCalendarUnavailableError as exc:
        assert "calendar down" in str(exc)
    else:
        raise AssertionError("strict calendar failure must be raised")


def test_legacy_backfill_keeps_hk_hold(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        backfill,
        "fetch_trade_cal",
        lambda pro, start, end, cache_dir, strict=False: _calendar("20260807"),
    )
    monkeypatch.setattr(
        backfill,
        "fetch_market_day",
        lambda pro, endpoint, day, cache_dir: calls.append(endpoint),
    )

    result = backfill.run_backfill(object(), "20260807", "20260807", tmp_path, max_workers=1)
    assert result["ok"] is True
    assert calls == list(backfill.LEGACY_ENDPOINTS)
    assert "hk_hold" in calls


@pytest.mark.parametrize(("start", "end"), [
    ("20260230", "20260301"),
    ("20260808", "20260807"),
])
def test_invalid_date_range_fails_before_io(
    tmp_path: Path, monkeypatch, start: str, end: str,
) -> None:
    monkeypatch.setattr(
        backfill, "fetch_trade_cal", lambda *args, **kwargs: pytest.fail("I/O attempted")
    )
    result = backfill.run_backfill(object(), start, end, tmp_path, strict_market=True)
    assert result["ok"] is False
    assert result["calendar_status"] == "failed"
    assert result["fatal_error"]["error_type"] == "ValueError"
    assert result["expected_pairs"] == result["completed_pairs"] == 0


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_max_workers_fails_before_io(
    tmp_path: Path, monkeypatch, workers,
) -> None:
    monkeypatch.setattr(
        backfill, "fetch_trade_cal", lambda *args, **kwargs: pytest.fail("I/O attempted")
    )
    result = backfill.run_backfill(
        object(), "20260807", "20260807", tmp_path,
        strict_market=True, max_workers=workers,
    )
    assert result["ok"] is False
    assert result["fatal_error"]["error_type"] == "ValueError"


@pytest.mark.parametrize("cal", [
    pd.DataFrame({"cal_date": ["20260807", "20260807"], "is_open": [1, 1]}),
    pd.DataFrame({"cal_date": ["20260230"], "is_open": [1]}),
    pd.DataFrame({"cal_date": ["20260808"], "is_open": [1]}),
    pd.DataFrame({"cal_date": ["20260807"], "is_open": [2]}),
    pd.DataFrame({"cal_date": ["20260807"], "is_open": ["unknown"]}),
])
def test_strict_trade_calendar_rejects_invalid_content(
    tmp_path: Path, monkeypatch, cal: pd.DataFrame,
) -> None:
    monkeypatch.setattr(backfill, "read_or_fetch", lambda path, pull: cal)
    with pytest.raises(backfill.TradeCalendarUnavailableError):
        backfill.fetch_trade_cal(
            object(), "20260807", "20260807", tmp_path, strict=True,
        )


def test_legacy_invalid_calendar_content_falls_back(
    tmp_path: Path, monkeypatch,
) -> None:
    cal = pd.DataFrame({"cal_date": ["20260807", "20260807"], "is_open": [1, 1]})
    monkeypatch.setattr(backfill, "read_or_fetch", lambda path, pull: cal)
    assert backfill.fetch_trade_cal(
        object(), "20260807", "20260807", tmp_path, strict=False,
    ) is None


def test_worker_exception_is_structured_fatal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        backfill,
        "fetch_trade_cal",
        lambda pro, start, end, cache_dir, strict=False: _calendar("20260807"),
    )
    monkeypatch.setattr(
        backfill, "_pull_day",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker crashed")),
    )
    result = backfill.run_backfill(
        object(), "20260807", "20260807", tmp_path,
        strict_market=True, max_workers=1,
    )
    assert result["ok"] is False
    assert result["fatal_error"] == {
        "error_type": "AssertionError", "error": "worker crashed",
    }


def test_main_passes_strict_env_to_tushare_factory(
    tmp_path: Path, monkeypatch,
) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(
        backfill, "tushare_pro",
        lambda **kwargs: seen.append(kwargs.get("strict_env")) or object(),
    )
    monkeypatch.setattr(
        backfill, "run_backfill",
        lambda *args, **kwargs: {
            "ok": True, "failed_pairs": [], "fatal_error": None,
        },
    )

    result = backfill.main(
        "20260807", "20260807", str(tmp_path),
        strict_market=True, strict_env=True,
    )

    assert result["ok"] is True
    assert seen == [True]


def test_strict_zero_open_days_is_valid_complete_range(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        backfill,
        "fetch_trade_cal",
        lambda pro, start, end, cache_dir, strict=False: pd.DataFrame({
            "cal_date": ["20260808", "20260809"], "is_open": [0, 0],
        }),
    )
    monkeypatch.setattr(
        backfill, "fetch_market_day", lambda *args, **kwargs: pytest.fail("pull attempted")
    )
    result = backfill.run_backfill(
        object(), "20260808", "20260809", tmp_path,
        strict_market=True, max_workers=1,
    )
    assert result["ok"] is True
    assert result["open_days"] == []
    assert result["expected_pairs"] == result["completed_pairs"] == 0
