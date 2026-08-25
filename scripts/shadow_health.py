"""Materialize the X-12 production-caliber monthly shadow health report.

Exit codes: 0=no review, 2=human review required, 1=input or I/O failure.
This command never changes production policy, holdings, or factor weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ashare_gauntlet.config import HOLDSCORE_DIR
from ashare_gauntlet.shadow_health import ShadowHealthError, build_shadow_report


DEFAULT_SOURCE = f"{HOLDSCORE_DIR}/composite_backtest.json"
DEFAULT_OUTPUT = f"{HOLDSCORE_DIR}/production_shadow_health.json"
CHINA_TZ = timezone(timedelta(hours=8))


def _load_source(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ShadowHealthError(f"{path} must contain a non-empty JSON list")
    return payload, hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="X-12 production-caliber monthly shadow return health")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 2:
            raise SystemExit(1) from exc
        raise

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    try:
        if source == output:
            raise ShadowHealthError("source and output must be different paths")
        rows, source_hash = _load_source(source)
        report = build_shadow_report(rows)
        report["generated_at"] = datetime.now(CHINA_TZ).isoformat(timespec="seconds")
        report["source"] = {"path": str(source), "sha256": source_hash}
        _atomic_write_json(output, report)
    except (OSError, UnicodeError, json.JSONDecodeError, ShadowHealthError,
            TypeError, ValueError) as exc:
        print(f"shadow_health failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    latest = next((row for row in reversed(report["observations"])
                   if row["status"] == "VALID"), None)
    latest_period = latest["period_end"] if latest else "none"
    print(
        f"X-12 shadow health: valid={report['valid_count']} invalid={report['invalid_count']} "
        f"latest={latest_period} negative_streak={report['current_negative_valid_streak']} "
        f"review_required={report['review_required']}"
    )
    print(f"-> {output}")
    raise SystemExit(2 if report["review_required"] else 0)


if __name__ == "__main__":
    main()
