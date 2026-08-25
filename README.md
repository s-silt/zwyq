# fxapk / A-share decision tools

## Daily brief and C2 review

The daily brief is read-only and consumes the `c2_state` projection embedded in
the dated decision snapshot:

```powershell
python -m scripts.daily_brief
python -m scripts.daily_brief --json
```

Run a validated monthly C2 review after the month-end evidence is ready:

```powershell
python -m scripts.c2_review --help
```

`WATCH` is informational. `REVIEW_BLOCKED_DATA` preserves the last valid review
date and does not advance the streak. `EXIT_RULE_C2_CONFIRMED` is an advisory
human action; it never edits holdings automatically. Missing state is shown as
`NOT_INITIALIZED`; unreadable or malformed state is shown as `UNAVAILABLE` /
degraded.

Daily brief exit codes:

- `0`: inputs are usable and there are no items requiring human handling.
- `1`: data, pipeline, snapshot, or C2 state is not trustworthy.
- `2`: the system is usable but a human review or action is required.
