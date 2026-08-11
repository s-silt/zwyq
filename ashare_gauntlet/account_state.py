"""账户状态安全基础 —— 纯函数模块(无IO/网络/stdout/import副作用)。

spec P0-1: legacy holdings 只读归一化、严格账户日期门禁、条件单 v2 结构验证、
独立 EOD 账户估值、MCP 统一接线。

全部纯函数,import阶段无IO/stdout。未知不解释为安全/0/已成交。
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

_DATE8 = re.compile(r"^\d{8}$")
_TS_CODE = re.compile(r"^\d{6}\.(?:SH|SZ)$", re.IGNORECASE)
_VALID_ORDER_STATUSES = frozenset({"active", "paused", "cancelled", "expired"})


# ── exception hierarchy ──

class AccountStateError(Exception):
    """账户状态异常基类。"""


class AccountSchemaError(AccountStateError):
    """账户 schema 异常(code=ACCOUNT_SCHEMA_ERROR)。"""


class AccountFreshnessError(AccountStateError):
    """账户日期新鲜度异常(code=ACCOUNT_FRESHNESS_ERROR)。"""


# ── freshness states ──

FRESHNESS_MISSING = "ACCOUNT_AS_OF_MISSING"
FRESHNESS_INVALID = "ACCOUNT_AS_OF_INVALID"
FRESHNESS_STALE   = "ACCOUNT_AS_OF_STALE"
FRESHNESS_FUTURE  = "ACCOUNT_AS_OF_FUTURE"
FRESHNESS_ALIGNED = "aligned"

# ── schema versions ──

SCHEMA_VERSION   = "account_state.v1"
LEGACY_SOURCE    = "legacy_unversioned"
EOD_SCHEMA       = "account_eod.v1"


# ── internal helpers ──

def _validate_date8(value: str) -> str:
    """校验 YYYYMMDD 为真实日期;非法格式/不可能日期 → AccountSchemaError。"""
    if not isinstance(value, str) or not _DATE8.fullmatch(value):
        raise AccountSchemaError(
            f"ACCOUNT_SCHEMA_ERROR: as_of 不是合法YYYYMMDD: {value!r}")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise AccountSchemaError(
            f"ACCOUNT_SCHEMA_ERROR: as_of 不是真实日期: {value!r}")
    return value


def _is_positive_integer(value: Any) -> bool:
    """正整数且非 bool；NaN/Inf 返回 False，不触发 int(inf)。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return value > 0 and float(value).is_integer()


def _is_finite_nonnegative(value: Any) -> bool:
    """有限非负数且非 bool。NaN/Inf → False。"""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and value >= 0)


def _is_finite_positive(value: Any) -> bool:
    """有限正数且非 bool。"""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and value > 0)


# ── public API ──

def normalize_account_state(
    raw: Any,
    expected_as_of: str | None = None,
    include_raw_orders: bool = False,
) -> dict[str, Any]:
    """将 legacy holdings 归一化为 account_state.v1 格式。

    不原地修改输入;不补造 position_id/entry_date/成交/lot/订单。
    未知字段不解释为安全/0。已知字段缺失/非法记入 missing/invalid。

    Args:
        raw: 原始 holdings JSON (dict)
        expected_as_of: 期望 as_of, 用于 freshness 判定(可选)
        include_raw_orders: True 时条件单输出 raw 字段(MCP 默认不暴露)

    Returns:
        归一化后的 account_state dict

    Raises:
        AccountSchemaError: 根/positions schema 非法
    """
    if not isinstance(raw, dict):
        raise AccountSchemaError(
            "ACCOUNT_SCHEMA_ERROR: holdings 根必须为 dict")

    positions_raw = raw.get("positions")
    if not isinstance(positions_raw, list):
        raise AccountSchemaError(
            "ACCOUNT_SCHEMA_ERROR: positions 必须为 list")

    missing: list[str] = []
    invalid: list[str] = []
    warnings: list[str] = []

    # ── as_of ──
    as_of_raw = raw.get("as_of")
    as_of: str | None = None
    if as_of_raw is None:
        pass  # 暂不记 missing——legacy 可能没这字段
    elif isinstance(as_of_raw, str) and _DATE8.fullmatch(as_of_raw):
        try:
            datetime.strptime(as_of_raw, "%Y%m%d")
            as_of = as_of_raw
        except ValueError:
            invalid.append("as_of")
            warnings.append(f"as_of={as_of_raw!r} 不是真实日期")
    else:
        invalid.append("as_of")

    source_schema = raw.get("source_schema", LEGACY_SOURCE)

    # ── cash ──
    cash_raw = raw.get("cash")
    if cash_raw is None:
        cash = None
        missing.append("cash")
    elif _is_finite_nonnegative(cash_raw):
        cash = float(cash_raw)
    else:
        cash = None
        invalid.append("cash")

    # ── positions ──
    seen_codes: set[str] = set()
    positions: list[dict[str, Any]] = []
    for i, p in enumerate(positions_raw):
        if not isinstance(p, dict):
            invalid.append(f"positions[{i}]")
            positions.append({"ts_code": None, "shares": 0})
            continue

        code = p.get("ts_code")
        if (not isinstance(code, str)
                or not _TS_CODE.fullmatch(code.upper())):
            invalid.append(f"positions[{i}].ts_code")
            positions.append(dict(p))
            continue

        if code in seen_codes:
            invalid.append(f"positions[{i}].ts_code(重复={code})")
        seen_codes.add(code)

        shares = p.get("shares")
        if not _is_positive_integer(shares):
            invalid.append(f"positions[{i}].shares")

        # cost / mv: 如存在需有限非负
        for fld in ("cost", "mv"):
            v = p.get(fld)
            if v is not None and v != "":
                if not _is_finite_nonnegative(v):
                    invalid.append(f"positions[{i}].{fld}")

        positions.append(dict(p))

    # ── conditional_orders ──
    raw_orders = raw.get("conditional_orders")
    orders_result = _normalize_conditional_orders(raw_orders, include_raw_orders)

    # ── data_status ──
    data_status = "complete" if not missing and not invalid else "incomplete"

    # ── freshness ──
    freshness = None
    if expected_as_of is not None:
        freshness = (FRESHNESS_INVALID if "as_of" in invalid
                     else classify_account_freshness(as_of, expected_as_of))

    # ── market_value / total_assets from manual mv (快照汇总,非EOD估值) ──
    mv_manual: float | None = None
    if cash is not None:
        mv_vals: list[float] = []
        for p in positions:
            v = p.get("mv")
            if _is_finite_nonnegative(v):
                mv_vals.append(float(v))
        if len(mv_vals) == len(positions_raw) and all(
            isinstance(x, dict) for x in positions_raw
        ):
            mv_manual = round(sum(mv_vals), 2)
        else:
            mv_manual = None

    total_manual = (
        round(mv_manual + cash, 2)
        if mv_manual is not None and cash is not None
        else None
    )

    # ── short_slot ──
    shorts = [
        p for p in positions
        if isinstance(p, dict) and p.get("bucket") == "短线"
    ]
    short_slot: dict[str, Any] = {
        "limit": 1,
        "count": len(shorts),
        "occupied": bool(shorts),
        "violation": len(shorts) > 1,
        "ts_codes": [
            str(p.get("ts_code"))
            for p in shorts
            if p.get("ts_code")
        ],
    }

    # ── industry_weights ──
    industry_weights = None
    if mv_manual is not None and total_manual is not None and total_manual > 0:
        industries: dict[str, float] = {}
        for p in positions:
            ind = str(p.get("industry") or "其他")
            v = p.get("mv")
            if _is_finite_nonnegative(v):
                industries[ind] = industries.get(ind, 0.0) + float(v)
        industry_weights = {
            key: {
                "market_value": round(value, 2),
                "weight": round(value / total_manual, 6),
            }
            for key, value in sorted(
                industries.items(), key=lambda item: -item[1]
            )
        }

    result: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "source_schema": source_schema,
        "data_status": data_status,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "warnings": warnings,
        "as_of": as_of,
        "cash": cash,
        "market_value": mv_manual,
        "total_assets": total_manual,
        "position_count": len(positions),
        "positions": positions,
        "short_position": shorts[0] if shorts else None,
        "short_slot": short_slot,
        "industry_weights": industry_weights,
        "conditional_orders": orders_result,
        "source": raw.get(
            "source",
            "data/holdings.json (manual single source of truth)",
        ),
    }
    if freshness is not None:
        result["freshness"] = freshness

    return result


def classify_account_freshness(
    account_as_of: str | None,
    expected_as_of: str | None,
) -> str:
    """判定账户 as_of 对期望日期的 freshness 状态。

    Returns 五个状态之一:
        aligned / ACCOUNT_AS_OF_MISSING / ACCOUNT_AS_OF_INVALID /
        ACCOUNT_AS_OF_STALE / ACCOUNT_AS_OF_FUTURE
    """
    if expected_as_of is None:
        return FRESHNESS_ALIGNED
    if account_as_of is None:
        return FRESHNESS_MISSING
    try:
        _validate_date8(account_as_of)
    except AccountSchemaError:
        return FRESHNESS_INVALID
    if account_as_of == expected_as_of:
        return FRESHNESS_ALIGNED
    if account_as_of < expected_as_of:
        return FRESHNESS_STALE
    return FRESHNESS_FUTURE


def require_account_as_of(
    account: dict[str, Any],
    expected_as_of: str,
) -> None:
    """要求账户 as_of 严格等于 expected_as_of;否则 AccountFreshnessError。

    不自动选日期、不修改源数据。
    """
    freshness = classify_account_freshness(
        account.get("as_of"), expected_as_of
    )
    if (freshness == FRESHNESS_MISSING
            and account.get("freshness") == FRESHNESS_INVALID
            and "as_of" in account.get("invalid_fields", [])):
        freshness = FRESHNESS_INVALID
    if freshness != FRESHNESS_ALIGNED:
        raise AccountFreshnessError(
            f"ACCOUNT_FRESHNESS_ERROR: 账户 as_of={account.get('as_of')!r} "
            f"≠ 期望日期 {expected_as_of} (状态={freshness})"
        )


def validate_conditional_orders_v2(
    value: dict[str, Any],
) -> dict[str, Any]:
    """验证 structured v2 条件单。

    要求: schema_version 整数 2、orders 列表。
    每单: 唯一非空 order_id、合法 ts_code、side BUY/SELL、
    condition.field 仅 close、operator <=/>=、price 正有限、
    shares 正整数、valid_from/valid_until 合法且顺序正确、
    status active/paused/cancelled/expired。

    合法 → verified + count;非法 → invalid + invalid_fields。
    不根据价格推断触发/成交;不修改源。
    未知版本不输出 verified。
    """
    if not isinstance(value, dict):
        return {
            "status": "invalid",
            "format": "structured_v2",
            "verified_count": None,
            "raw_present": True,
            "invalid_fields": ["根不是 dict"],
        }

    sv = value.get("schema_version")
    if not isinstance(sv, int) or sv != 2:
        sv_str = repr(sv)
        return {
            "status": "unverified",
            "format": f"unknown_schema_version_{sv_str}",
            "verified_count": None,
            "raw_present": True,
            "invalid_fields": [f"schema_version={sv_str}"],
        }

    orders = value.get("orders")
    if not isinstance(orders, list):
        return {
            "status": "invalid",
            "format": "structured_v2",
            "verified_count": None,
            "raw_present": True,
            "invalid_fields": ["orders: 不是 list"],
        }

    invalid_fields: list[str] = []
    seen_ids: set[str] = set()
    valid_count = 0

    for i, order in enumerate(orders):
        prefix = f"orders[{i}]"
        if not isinstance(order, dict):
            invalid_fields.append(f"{prefix}: 不是 dict")
            continue

        order_invalid_count = len(invalid_fields)

        # order_id
        oid = order.get("order_id")
        if not isinstance(oid, str) or not oid:
            invalid_fields.append(f"{prefix}.order_id: 缺失或空")
        elif oid in seen_ids:
            invalid_fields.append(f"{prefix}.order_id: 重复={oid}")
        else:
            seen_ids.add(oid)

        # ts_code
        code = order.get("ts_code")
        if not isinstance(code, str) or not _TS_CODE.fullmatch(str(code).upper()):
            invalid_fields.append(f"{prefix}.ts_code: 无效={code!r}")

        # side
        side = order.get("side")
        if side not in ("BUY", "SELL"):
            invalid_fields.append(f"{prefix}.side: 无效={side!r}")

        # condition
        cond = order.get("condition")
        if isinstance(cond, dict):
            field = cond.get("field")
            if field != "close":
                invalid_fields.append(
                    f"{prefix}.condition.field: 仅支持close, 得到={field!r}"
                )
            op = cond.get("operator")
            if op not in (">=", "<="):
                invalid_fields.append(
                    f"{prefix}.condition.operator: 无效={op!r}"
                )
        else:
            invalid_fields.append(f"{prefix}.condition: 不是 dict")

        # price
        price = order.get("price")
        if not _is_finite_positive(price):
            invalid_fields.append(
                f"{prefix}.price: 非正有限数={price!r}"
            )

        # shares
        shares = order.get("shares")
        if not _is_positive_integer(shares):
            invalid_fields.append(
                f"{prefix}.shares: 非正整数={shares!r}"
            )

        # valid_from / valid_until
        vf = order.get("valid_from")
        vu = order.get("valid_until")
        vf_valid = True
        vu_valid = True
        vf_str: str | None = None
        vu_str: str | None = None
        if vf is None or vf == "":
            invalid_fields.append(f"{prefix}.valid_from: 缺失")
            vf_valid = False
        else:
            try:
                vf_str = _validate_date8(str(vf))
            except (AccountSchemaError, TypeError):
                invalid_fields.append(f"{prefix}.valid_from: 无效={vf!r}")
                vf_valid = False
        if vu is None or vu == "":
            invalid_fields.append(f"{prefix}.valid_until: 缺失")
            vu_valid = False
        else:
            try:
                vu_str = _validate_date8(str(vu))
            except (AccountSchemaError, TypeError):
                invalid_fields.append(f"{prefix}.valid_until: 无效={vu!r}")
                vu_valid = False
        if vf_valid and vu_valid and vf_str and vu_str and vf_str > vu_str:
            invalid_fields.append(
                f"{prefix}: valid_from({vf}) > valid_until({vu})"
            )

        # status
        st = order.get("status")
        if st not in _VALID_ORDER_STATUSES:
            invalid_fields.append(f"{prefix}.status: 无效={st!r}")

        # 该单无新增 invalid → valid
        if len(invalid_fields) == order_invalid_count:
            valid_count += 1

    if invalid_fields:
        return {
            "status": "invalid",
            "format": "structured_v2",
            "verified_count": valid_count,
            "raw_present": True,
            "invalid_fields": invalid_fields,
        }
    return {
        "status": "verified",
        "format": "structured_v2",
        "verified_count": valid_count,
        "raw_present": True,
    }


def build_eod_account_valuation(
    account: dict[str, Any],
    as_of: str,
    position_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建 EOD 账户估值:严格 shares × 当日 EOD close,不使用人工 mv。

    Args:
        account: 归一化后的账户
        as_of: 估值日期 YYYYMMDD
        position_records: build_position_record 产出的持仓估值记录

    Returns:
        valuation dict:
          status(complete|incomplete), basis=shares_x_eod_close,
          market_value, cash, total_assets, position_count, valued_position_count
        任一持仓无有效当日行情 → market_value/total_assets=None, status=incomplete
    """
    _validate_date8(as_of)
    positions = account.get("positions", [])
    if not isinstance(positions, list):
        raise AccountSchemaError(
            "ACCOUNT_SCHEMA_ERROR: positions 必须为 list")
    if not positions:
        cash = account.get("cash")
        complete = _is_finite_nonnegative(cash)
        return {
            "status": "complete" if complete else "incomplete",
            "basis": "shares_x_eod_close",
            "market_value": 0.0 if complete else None,
            "cash": float(cash) if complete else None,
            "total_assets": float(cash) if complete else None,
            "position_count": 0,
            "valued_position_count": 0,
        }

    # ts_code → close；重复或非有限记录保持不可估值。
    close_map: dict[str, float | None] = {}
    duplicate_records: set[str] = set()
    for rec in position_records:
        if not isinstance(rec, dict):
            continue
        code = rec.get("ts_code")
        if not code:
            continue
        code = str(code)
        if code in close_map:
            duplicate_records.add(code)
            close_map[code] = None
            continue
        close = rec.get("close")
        if rec.get("error") is None and _is_finite_positive(close):
            close_map[code] = float(close)
        else:
            close_map[code] = None

    mv_sum = 0.0
    valued_count = 0
    all_valued = True
    seen_positions: set[str] = set()

    for p in positions:
        code = str(p.get("ts_code", ""))
        shares = p.get("shares")
        if (not code or code in seen_positions or code in duplicate_records
                or not _is_positive_integer(shares)):
            all_valued = False
            continue
        seen_positions.add(code)
        close = close_map.get(code)
        if close is None:
            all_valued = False
            continue
        mv_sum += float(int(shares)) * close
        valued_count += 1

    market_value = round(mv_sum, 2) if all_valued and valued_count > 0 else None
    cash = account.get("cash")
    total_assets = (
        round(market_value + cash, 2)
        if market_value is not None and _is_finite_nonnegative(cash)
        else None
    )

    return {
        "status": "complete" if all_valued and total_assets is not None else "incomplete",
        "basis": "shares_x_eod_close",
        "market_value": market_value,
        "cash": float(cash) if _is_finite_nonnegative(cash) else None,
        "total_assets": total_assets,
        "position_count": len(positions),
        "valued_position_count": valued_count,
    }


# ── internal: conditional_orders normalization ──

def _normalize_conditional_orders(
    value: Any,
    include_raw: bool,
) -> dict[str, Any]:
    """normalize conditional_orders 字段。

    legacy string → unverified, format=legacy_free_text
    v2 dict → 经 validate_conditional_orders_v2
    其它 → unverified
    """
    if value in (None, ""):
        return {
            "status": "missing",
            "format": None,
            "verified_count": None,
            "raw_present": False,
        }

    # structured v2: dict with schema_version integer
    if isinstance(value, dict):
        sv = value.get("schema_version")
        if isinstance(sv, int) and sv == 2:
            result = validate_conditional_orders_v2(value)
            if include_raw:
                result["raw"] = value
            return result
        else:
            sv_str = repr(sv)
            return {
                "status": "unverified",
                "format": f"unknown_schema_version_{sv_str}",
                "verified_count": None,
                "raw_present": True,
                "invalid_fields": [f"schema_version={sv_str}"],
            }

    # legacy: string or other
    base: dict[str, Any] = {
        "status": "unverified",
        "format": "legacy_free_text",
        "verified_count": None,
        "raw_present": True,
    }
    if include_raw:
        base["raw"] = value
    return base
