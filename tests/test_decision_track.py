"""decision_track CLI 的离线 IO、边界和原子写测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_gauntlet.decision_evaluation import DecisionEvaluationError
from scripts import decision_track as dt


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _snapshot(date="20260101"):
    return {
        "as_of": date,
        "generated_at": f"{date[:4]}-{date[4:6]}-{date[6:]}T18:00:00+08:00",
        "factor_snapshot": f"data/holdscore/{date}_factor.json",
        "policy_version": "1",
        "entry_model_version": "research-only",
        "data_status": "complete",
        "decisions": [{
            "ts_code": "600001.SH", "name": "甲", "state": "BUY",
            "reason_codes": ["D10", "TIER_GREEN", "FACTCHECK_CLEAR"],
            "evidence": {"decile": 10, "score": 0.9},
            "execution": {"eligible_from": "NEXT_TRADING_DAY", "max_entry_price": None,
                          "target_weight": 0.1, "shares": 100},
            "invalidations": ["RISK_RED_FLAG"],
        }],
    }


def _fixture(root: Path) -> None:
    boundary = _snapshot("20251231")
    boundary["decisions"][0]["state"] = "WAIT"
    boundary["decisions"][0]["reason_codes"] = ["FACTCHECK_REQUIRED"]
    boundary["decisions"][0]["execution"]["shares"] = 0
    _write_json(root / "data/decisions/20251231_buy_decisions.json", boundary)
    _write_json(root / "data/holdscore/20251231_factor.json",
                [{"ts_code": "600001.SH", "name": "甲", "decile": 10, "score": 0.9}])
    _write_json(root / "data/decisions/20260101_buy_decisions.json", _snapshot())
    _write_json(root / "data/holdscore/20260101_factor.json",
                [{"ts_code": "600001.SH", "name": "甲", "decile": 10, "score": 0.9}])
    for date, price in (("20260101", 9.0), ("20260102", 10.0), ("20260105", 11.0)):
        daily = pd.DataFrame([
            {"ts_code": "600001.SH", "trade_date": date, "open": price,
             "high": price + 0.2, "low": price - 0.2, "close": price},
        ])
        adj = pd.DataFrame([{"ts_code": "600001.SH", "trade_date": date,
                             "adj_factor": 1.0}])
        basic = pd.DataFrame([{"ts_code": "600001.SH", "trade_date": date,
                               "total_mv": 100000.0}])
        limit = pd.DataFrame([{"ts_code": "600001.SH", "trade_date": date,
                               "up_limit": 99.0, "down_limit": 0.1}])
        for endpoint, frame in (("daily", daily), ("adj_factor", adj),
                                ("daily_basic", basic), ("stk_limit", limit)):
            path = root / f"data/cache/{endpoint}/{date}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)


def test_build_report_is_offline_and_does_not_read_current_manual_state(tmp_path, monkeypatch):
    _fixture(tmp_path)
    for name in ("holdings.json", "trading_policy.json", "factcheck_overrides.json"):
        (tmp_path / "data" / name).write_text("this must not be read", encoding="utf-8")
    seen = []
    real_json = dt._json

    def recording_json(path):
        seen.append(Path(path).name)
        return real_json(path)

    monkeypatch.setattr(dt, "_json", recording_json)
    report = dt.build_report(tmp_path, horizons=(1,))
    assert report["schema"] == "decision_chain_evaluation.v1"
    assert report["scope"]["actual_execution"] is False
    assert report["scope"]["production_policy_changed"] is False
    assert report["coverage"]["buy_episode_count"] == 1
    assert report["metrics"]["1"]["resolved_count"] == 1
    assert not {"holdings.json", "trading_policy.json", "factcheck_overrides.json"} & set(seen)
    encoded = json.dumps(report, ensure_ascii=False)
    assert "this must not be read" not in encoded


def test_invalid_snapshot_is_reported_without_market_or_manual_inputs(tmp_path):
    broken = _snapshot()
    broken["decisions"][0]["reason_codes"] = ["FACTCHECK_REQUIRED"]
    _write_json(tmp_path / "data/decisions/20260101_buy_decisions.json", broken)
    _write_json(tmp_path / "data/holdscore/20260101_factor.json",
                [{"ts_code": "600001.SH", "decile": 10}])
    report = dt.build_report(tmp_path, horizons=(1,))
    assert report["coverage"]["snapshot_invalid"] == 1
    assert report["events"] == []
    assert "FACTCHECK_CLEAR" in " ".join(report["snapshot_audits"][0]["errors"])


def test_atomic_write_failure_does_not_overwrite_existing_report(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text('{"old": true}', encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("serialize failed")

    monkeypatch.setattr(dt.json, "dump", fail)
    with pytest.raises(RuntimeError, match="serialize failed"):
        dt.atomic_write_json(path, {"new": True})
    assert path.read_text(encoding="utf-8") == '{"old": true}'
    assert not list(tmp_path.glob("*.tmp"))


def test_main_writes_deterministic_report_in_tmp_root(tmp_path):
    _fixture(tmp_path)
    output = "data/holdscore/custom_eval.json"
    dt.main(["--root", str(tmp_path), "--horizons", "1", "--output", output])
    first = (tmp_path / output).read_text(encoding="utf-8")
    dt.main(["--root", str(tmp_path), "--horizons", "1", "--output", output])
    assert (tmp_path / output).read_text(encoding="utf-8") == first


def test_output_path_cannot_escape_project_root(tmp_path):
    _fixture(tmp_path)
    with pytest.raises(DecisionEvaluationError, match="逃出项目根"):
        dt.main(["--root", str(tmp_path), "--horizons", "1", "--output", "../leak.json"])


def test_main_rejects_impossible_calendar_date(tmp_path):
    with pytest.raises(SystemExit):
        dt.main(["--root", str(tmp_path), "--start", "20260230"])


def test_start_filter_uses_prior_snapshot_as_episode_boundary(tmp_path):
    _fixture(tmp_path)
    report = dt.build_report(tmp_path, start="20260101", horizons=(1,))
    assert report["coverage"]["buy_episode_count"] == 1
    assert report["coverage"]["left_censored_buy_count"] == 0
    assert report["snapshot_audits"][0]["file_date"] == "20260101"


def test_four_core_date_mismatch_fails_loud(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "data/cache/daily_basic/20260102.parquet").unlink()
    with pytest.raises(DecisionEvaluationError, match="四核心 EOD 日期集合不一致"):
        dt.build_report(tmp_path, horizons=(1,))


def _write_trade_cal(root: Path, is_open: list[int]) -> None:
    dates = ["20260101", "20260102", "20260103", "20260104", "20260105"]
    cal = pd.DataFrame({"cal_date": dates, "is_open": is_open})
    path = root / "data/cache/trade_cal/20260101_20260105.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(path, index=False)


def test_calendar_proven_missing_open_day_fails_loud(tmp_path):
    """codex P1-2a:归档日历完整且证明某开市日缺全部四核心分区 → 不得继续评估。"""
    _fixture(tmp_path)  # 分区日 0101/0102/0105
    _write_trade_cal(tmp_path, [1, 1, 0, 1, 1])  # 0104 开市但无分区
    with pytest.raises(DecisionEvaluationError, match="四核心分区"):
        dt.build_report(tmp_path, horizons=(1,))


def test_partial_calendar_marks_t_plus_one_unverified(tmp_path):
    """codex P1-2b:日历归档不完整时,T+1 依据须逐窗口标 unverified。"""
    _fixture(tmp_path)  # 无 trade_cal 归档
    report = dt.build_report(tmp_path, horizons=(1,))
    assert report["calendar_coverage"]["status"] != "complete"
    assert report["metrics"]["1"]["t_plus_one_basis"] == "partition_dates_only_unverified"


def test_complete_calendar_marks_t_plus_one_verified(tmp_path):
    _fixture(tmp_path)
    _write_trade_cal(tmp_path, [1, 1, 0, 0, 1])  # 开市日与分区日一致
    report = dt.build_report(tmp_path, horizons=(1,))
    assert report["calendar_coverage"]["status"] == "complete"
    assert (report["metrics"]["1"]["t_plus_one_basis"]
            == "verified_against_complete_calendar")


def test_snapshot_gap_over_known_trade_day_breaks_episode(tmp_path):
    """codex P1-3:0102 是已知交易日却无快照,0105 的 BUY 不得并入 0101 episode。

    突变锁:把 extract_buy_episodes 的 known_trade_days 传参退回不传,
    本测试必须变红(0105 被当延续,left_censored=0)。
    """
    _fixture(tmp_path)  # 20251231 WAIT + 20260101 BUY;分区 0101/0102/0105
    _write_json(tmp_path / "data/decisions/20260105_buy_decisions.json",
                _snapshot("20260105"))
    _write_json(tmp_path / "data/holdscore/20260105_factor.json",
                [{"ts_code": "600001.SH", "name": "甲", "decile": 10, "score": 0.9}])
    report = dt.build_report(tmp_path, horizons=(1,))
    assert report["coverage"]["left_censored_buy_count"] == 1
    assert report["coverage"]["buy_episode_count"] == 1
