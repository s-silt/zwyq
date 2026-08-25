from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.shadow_health import main


def _rows(net_negative: bool = True) -> list[dict[str, float | str]]:
    portfolio = 0.0 if net_negative else 0.03
    return [
        {"date": date, "ret_PROD": portfolio, "mkt_fwd": 0.01,
         "TO_PROD": 0.5, "cost_rt": 0.004}
        for date in ("20260130", "20260227", "20260331")
    ]


def _write_source(path: Path, rows: list[dict[str, object]]) -> str:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_writes_hashed_report_and_exits_two_for_review(tmp_path: Path) -> None:
    source = tmp_path / "composite.json"
    output = tmp_path / "shadow.json"
    expected_hash = _write_source(source, _rows())

    with pytest.raises(SystemExit) as exc:
        main(["--source", str(source), "--output", str(output)])

    assert exc.value.code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source"] == {
        "path": str(source.resolve()),
        "sha256": expected_hash,
    }
    assert report["generated_at"].endswith("+08:00")
    assert report["review_required"] is True


def test_cli_exits_zero_when_review_is_not_required(tmp_path: Path) -> None:
    source = tmp_path / "composite.json"
    output = tmp_path / "shadow.json"
    _write_source(source, _rows(net_negative=False))

    with pytest.raises(SystemExit) as exc:
        main(["--source", str(source), "--output", str(output)])

    assert exc.value.code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["review_required"] is False


@pytest.mark.parametrize("content", ["{bad json", "[]", "{}"])
def test_bad_source_does_not_replace_existing_report(
        tmp_path: Path, content: str) -> None:
    source = tmp_path / "composite.json"
    output = tmp_path / "shadow.json"
    source.write_text(content, encoding="utf-8")
    output.write_text("preserve-me", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["--source", str(source), "--output", str(output)])

    assert exc.value.code == 1
    assert output.read_text(encoding="utf-8") == "preserve-me"


def test_cli_refuses_to_replace_its_source(tmp_path: Path) -> None:
    source = tmp_path / "composite.json"
    _write_source(source, _rows())
    original = source.read_bytes()

    with pytest.raises(SystemExit) as exc:
        main(["--source", str(source), "--output", str(source)])

    assert exc.value.code == 1
    assert source.read_bytes() == original


def test_cli_argument_errors_exit_one() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--unknown"])

    assert exc.value.code == 1
