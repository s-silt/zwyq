"""门禁证据体检 CLI(只读 + 可冻结基线):准入证据有没有在静默变旧?

用法(季度节律,政策评审 Q4 的 D 方案):
1. 先复跑权威读数(慢,数分钟;不在 eod_ops 里,按季手动或排期跑):
     python -m scripts.factor_backtest
     python -m scripts.composite_backtest
2. 体检并与冻结基线比对:
     python -m scripts.gate_check
3. 首次或经人工复审后重新冻结基线:
     python -m scripts.gate_check --freeze

退出码:0=证据健康;2=有 DRIFT/DEGRADED 需人工复审;1=读数缺失/结构失败。
**本工具只报不改**:任何等级都不会自动改 composite、权重或门槛——那是研究治理决定。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

from ashare_gauntlet.config import HOLDSCORE_DIR
from ashare_gauntlet.gate_health import build_report, compare_to_baseline

DETAIL = f"{HOLDSCORE_DIR}/factor_ic_backtest.json"
COMPOSITE = f"{HOLDSCORE_DIR}/composite_backtest.json"
BASELINE = f"{HOLDSCORE_DIR}/gate_baseline.json"


def _load(path: str, what: str, required: bool = True):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        if required:
            raise SystemExit(f"{path} 不存在——{what};先复跑对应回测(见本模块 docstring)")
        return None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} 不是合法 JSON: {exc}")


def _atomic_dump(path: str, payload: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix=".tmp_gate_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv: "list[str] | None" = None) -> None:
    ap = argparse.ArgumentParser(description="门禁证据体检(五门 + 组合 t vs 冻结基线)")
    ap.add_argument("--freeze", action="store_true",
                    help="把本次体检结果冻结为新基线(首次建立,或人工复审结案后)")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON")
    ap.add_argument("--detail", default=DETAIL)
    ap.add_argument("--composite", default=COMPOSITE)
    ap.add_argument("--baseline", default=BASELINE)
    a = ap.parse_args(argv)

    rows = _load(a.detail, "因子回测读数")
    comp_rows = _load(a.composite, "组合回测读数", required=False)
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{a.detail} 应为非空列表")
    now = datetime.now(timezone(timedelta(hours=8)))
    report = build_report(rows, comp_rows if isinstance(comp_rows, list) else None,
                          as_of=now.strftime("%Y%m%d"))

    if a.freeze:
        payload = {**report, "frozen_at": now.isoformat(),
                   "note": "门禁证据基线:后续体检与本读数比对;经人工复审结案后才重新冻结"}
        _atomic_dump(a.baseline, payload)
        print(f"已冻结基线 → {a.baseline}")
        print(f"  样本 N={report['sample']['n']} {report['sample']['first']}→{report['sample']['last']}")
        for f in report["factors"]:
            print(f"  {f['factor']:>5}: {f['status']} NW t={f.get('nw_t')}")
        print(f"  composite: NW t={report['composite'].get('nw_t')}")
        raise SystemExit(0)

    baseline = _load(a.baseline, "门禁基线", required=False)
    findings = compare_to_baseline(report, baseline or {}) if isinstance(baseline, dict) else []

    if a.json:
        print(json.dumps({"report": report, "baseline_present": bool(baseline),
                          "findings": findings}, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        s = report["sample"]
        print(f"=== 门禁证据体检 === N={s['n']} {s['first']}→{s['last']}")
        for f in report["factors"]:
            mark = "✅" if f["status"] == "PASS" else "❌"
            line = f"  {mark} {f['factor']:>5} {f['status']}"
            if f.get("nw_t") is not None:
                line += (f" | NW t={f['nw_t']} 真实净={f['real_net']*100:+.2f}%/期"
                         f" 多头腿={f['leg_net']*100:+.2f}%/期"
                         f" LOYO最弱|t|={f['loyo_min_abs_t']}"
                         f" 涨/跌市 IC {f['up_ic']:+.3f}/{f['down_ic']:+.3f}")
            print(line)
            for r in f.get("reasons", []):
                print(f"        - {r}")
        c = report["composite"]
        print(f"  组合 {c.get('port')}: NW t={c.get('nw_t')}(毛超额口径,与 methodology §10 同)"
              f" 毛={c.get('gross_mean_pct')}%/期 净={c.get('net_mean_pct')}%/期"
              f" 净t={c.get('nw_t_net')} (N={c.get('n')})"
              + (f"  {c.get('note')}" if c.get("note") else ""))
        if not baseline:
            print("\n(尚无冻结基线——先跑 --freeze 建立;仅本次读数无法判断是否退化)")
        elif not findings:
            print("\n✅ 与基线一致:准入证据未见退化")
        else:
            print(f"\n⚠ {len(findings)} 条需人工复审:")
            for f in findings:
                print(f"  [{f['level']}] {f['target']} {f['issue']}: {f['detail']}")
            print("\n(本工具只报不改;复审时请拆分:数据质量/实现偏差/成本口径/"
                  "因子暴露漂移/风格逆风/统计失效——勿把风格逆风直接当因子死亡)")
    raise SystemExit(2 if findings else 0)


if __name__ == "__main__":
    main()
