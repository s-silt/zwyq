"""生产候选资格与硬否决(spec §5 + X-14)——从 factor snapshot 行构造 BUY 资格判定。

设计约束(全部承自 docs/superpowers/specs/2026-07-19-actionable-buy-decisions-design.md):
- 生产候选池 = **当期 D10 ∪ B8 带保留成员**(X-14,用户认可 2026-08-31 采纳):
  D10 之外,上期成员仍处 D8+ 带内的也保留候选资格——组合级实验(N=139)显示
  排名带缓冲把换手 41%→17% 的同时净超额 +0.29%→+0.36%、年胜率 62%→69%。
  b8_band 由 buy_list 按持久化 B8 状态(唯一真相源)注入,本层只消费不计算;
  🟡 最高 WAIT;风险层只否决不加分;
- 第四关强制:BUY 必须有未过期 verdict=clear 的人工 factcheck 覆盖
  (data/factcheck_overrides.json,过期自动回 WAIT);
- reason code 顺序=固定检查序,运行间不变(下游测试与 CLI 依赖);
- 字段缺失 fail-loud(KeyError 上抛,不许把残缺行判成可买)。
"""
from __future__ import annotations

import re
from datetime import datetime

from ashare_gauntlet.screen import board_of

_DATE8 = re.compile(r"^\d{8}$")

MAIN = ("沪主板", "深主板")

# 固定检查序(spec §12:reason code 完整且顺序确定)
# NOT_D10 语义(X-14 后)= 不在生产候选池(既非当期 D10,也非 B8 带保留成员)
_CHECKS = ("NOT_MAIN_BOARD", "ST_NAME", "NOT_D10", "TIER_NOT_GREEN",
           "SPEC_CROWD", "SPIKE_LIMIT", "POLLUTION_PENDING_FACTCHECK",
           "FACTCHECK_AFTER_AS_OF", "GOVERNANCE_RED", "FACTCHECK_EXPIRED",
           "FACTCHECK_REQUIRED")

# **硬否决码**(权威单一来源):写人工 verdict=clear 也解不开的否决项。
# "唯一未决项=fact-check"的候选 = state==WAIT ∧ FACTCHECK_REQUIRED ∧ 无任何硬否决码。
# 此前 eod_ops / factcheck_probe / daily_brief 各自手写一份只含三码的排除表,
# 漏掉 SPEC_CROWD(🎰投机拥挤)与 SPIKE_LIMIT(⚡近5日触涨停),把这类票呈现成
# "只差一步就能买",诱导用户白跑 fact-check、甚至绕开机器状态手动买入(跨层审计)。
HARD_VETO_CODES = frozenset(c for c in _CHECKS if not c.startswith("FACTCHECK_"))


def _date8(value: object, field: str) -> str:
    if not isinstance(value, str) or not _DATE8.fullmatch(value):
        raise ValueError(f"{field} 必须是真实 YYYYMMDD,得到 {value!r}")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{field} 必须是真实 YYYYMMDD,得到 {value!r}") from exc
    return value


def override_status(override: "dict | None", as_of: str) -> str:
    """人工 factcheck 覆盖状态:clear / red / expired / missing。

    verdict 白名单 {clear, red},其余值(笔误/pending/大小写)fail-loud——
    第四关是 BUY 的硬闸,把未知当安全=红灯笔误直接放行(对抗审查 P1)。
    """
    decision_as_of = _date8(as_of, "decision as_of")
    if override is None:
        return "missing"
    if not isinstance(override, dict):
        raise ValueError("factcheck 覆盖必须是对象")
    required = ("verdict", "as_of", "expires_on")
    missing = [key for key in required if key not in override]
    if missing:
        raise ValueError(f"factcheck 覆盖缺字段 {missing}")
    v = override["verdict"]
    if not isinstance(v, str) or v not in ("clear", "red"):
        raise ValueError(f"factcheck 覆盖 verdict={v!r} 不在白名单 {{clear,red}}——请修正覆盖文件")
    override_as_of = _date8(override["as_of"], "factcheck as_of")
    expires_on = _date8(override["expires_on"], "factcheck expires_on")
    if expires_on < override_as_of:
        raise ValueError("factcheck expires_on 不能早于 factcheck as_of")
    if override_as_of > decision_as_of:
        return "future"
    if v == "red":
        return "red"
    if expires_on < decision_as_of:
        return "expired"
    return "clear"


def candidate_assessment(row: dict, override: "dict | None", as_of: str) -> dict:
    """单行资格判定 → {ts_code, eligible_buy, reason_codes, governance_red}。

    row 必须含 ts_code/name/decile/tier/spec_crowd/spike_limit(缺失 KeyError
    fail-loud);poll_mark 可选(结构化污染标记,来自 R6 探针口径,缺省=未标记);
    b8_band 可选(X-14 带保留成员标记,buy_list 注入;缺省=非带成员,行为同旧口径)。
    """
    ts, name = str(row["ts_code"]), str(row["name"])
    ov = override_status(override, as_of)
    hits: set[str] = set()
    if board_of(ts) not in MAIN:
        hits.add("NOT_MAIN_BOARD")
    if "ST" in name:
        hits.add("ST_NAME")
    if row["decile"] != 10 and not (bool(row.get("b8_band", False)) and row["decile"] in (8, 9)):
        hits.add("NOT_D10")
    if row["tier"] != "🟢":
        hits.add("TIER_NOT_GREEN")
    if bool(row["spec_crowd"]):
        hits.add("SPEC_CROWD")
    if bool(row["spike_limit"]):
        hits.add("SPIKE_LIMIT")
    if bool(row.get("poll_mark", False)):
        hits.add("POLLUTION_PENDING_FACTCHECK")
    if ov == "future":
        hits.add("FACTCHECK_AFTER_AS_OF")
    elif ov == "red":
        hits.add("GOVERNANCE_RED")
    elif ov == "expired":
        hits.add("FACTCHECK_EXPIRED")
    elif ov == "missing":
        hits.add("FACTCHECK_REQUIRED")
    codes = [c for c in _CHECKS if c in hits]
    eligible = not codes
    if eligible:
        # 池内来源码分开报:当期 D10 vs B8 带保留(X-14)——审计/复核据此区分入场路径
        codes = ["D10" if row["decile"] == 10 else "B8_BAND",
                 "TIER_GREEN", "FACTCHECK_CLEAR"]
    return {"ts_code": ts, "name": name, "industry": row.get("industry", "其他"),
            "score": row.get("score"), "last": row.get("last"),
            "decile": row["decile"], "spec_crowd": bool(row["spec_crowd"]),
            "spike_limit": bool(row["spike_limit"]),
            "eligible_buy": eligible, "reason_codes": codes,
            "governance_red": ov == "red"}
