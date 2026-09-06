"""tests for ashare_gauntlet.account_state —— P0-1 账户状态安全基础。

覆盖: legacy归一化、不修改输入、异常字段、重复代码、0与未知、
全部freshness状态、legacy条件单默认无raw和显式raw、
v2合法/非法、EOD聚合及缺行情。
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from ashare_gauntlet.account_state import (
    FRESHNESS_ALIGNED,
    FRESHNESS_FUTURE,
    FRESHNESS_INVALID,
    FRESHNESS_MISSING,
    FRESHNESS_STALE,
    AccountFreshnessError,
    AccountSchemaError,
    build_eod_account_valuation,
    classify_account_freshness,
    normalize_account_state,
    require_account_as_of,
    validate_conditional_orders_v2,
)


# ── helpers ──

def _legacy_holdings(**kw) -> dict:
    base: dict = {
        "as_of": "20260807",
        "cash": 50000.0,
        "positions": [
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "industry": "银行",
                "shares": 1000,
                "cost": 12.5,
                "mv": 12500.0,
                "bucket": "长线",
            },
            {
                "ts_code": "000002.SZ",
                "name": "万科A",
                "industry": "地产",
                "shares": 500,
                "cost": 15.0,
                "mv": 7500.0,
                "bucket": "短线",
            },
        ],
    }
    base.update(kw)
    return base


def _v2_order(**kw) -> dict:
    base: dict = {
        "schema_version": 2,
        "orders": [
            {
                "order_id": "ord-001",
                "ts_code": "000001.SZ",
                "side": "BUY",
                "condition": {"field": "close", "operator": "<="},
                "price": 10.5,
                "shares": 100,
                "valid_from": "20260801",
                "valid_until": "20260831",
                "status": "active",
            },
        ],
    }
    base.update(kw)
    return base


# ── normalize_account_state: legacy ──

def test_normalize_legacy_basic():
    raw = _legacy_holdings()
    result = normalize_account_state(raw)
    assert result["schema"] == "account_state.v1"
    assert result["source_schema"] == "legacy_unversioned"
    assert result["as_of"] == "20260807"
    assert result["cash"] == 50000.0
    assert result["data_status"] == "complete"
    assert result["position_count"] == 2
    assert result["total_assets"] == 20000.0 + 50000.0  # 12500 + 7500 + 50000
    assert result["short_slot"]["count"] == 1
    assert result["short_slot"]["violation"] is False
    assert result["industry_weights"]["银行"]["weight"] == pytest.approx(12500 / 70000, abs=1e-6)
    assert result["conditional_orders"]["status"] == "missing"
    assert "raw" not in result["conditional_orders"]


def test_normalize_does_not_mutate_input():
    raw = _legacy_holdings()
    original = copy.deepcopy(raw)
    normalize_account_state(raw)
    assert raw == original


def test_normalize_rejects_non_dict_root():
    with pytest.raises(AccountSchemaError, match="根必须为 dict"):
        normalize_account_state([])
    with pytest.raises(AccountSchemaError, match="根必须为 dict"):
        normalize_account_state("not dict")


def test_normalize_rejects_non_list_positions():
    with pytest.raises(AccountSchemaError, match="positions 必须为 list"):
        normalize_account_state({"positions": "not list"})


def test_normalize_duplicate_ts_code():
    raw = _legacy_holdings(positions=[
        {"ts_code": "000001.SZ", "shares": 100, "mv": 1000, "industry": "银行"},
        {"ts_code": "000001.SZ", "shares": 200, "mv": 2000, "industry": "银行"},
    ])
    result = normalize_account_state(raw)
    assert "positions[1].ts_code(重复=000001.SZ)" in result["invalid_fields"]


def test_normalize_bad_shares():
    raw = _legacy_holdings(positions=[
        {"ts_code": "000001.SZ", "shares": True, "mv": 1000, "industry": "银行"},
    ])
    result = normalize_account_state(raw)
    assert "positions[0].shares" in result["invalid_fields"]

    raw2 = _legacy_holdings(positions=[
        {"ts_code": "000001.SZ", "shares": 100.5, "mv": 1000, "industry": "银行"},
    ])
    result2 = normalize_account_state(raw2)
    assert any("shares" in f for f in result2["invalid_fields"])

    raw3 = _legacy_holdings(positions=[
        {"ts_code": "000001.SZ", "shares": 0, "mv": 1000, "industry": "银行"},
    ])
    result3 = normalize_account_state(raw3)
    assert any("shares" in f for f in result3["invalid_fields"])


def test_normalize_missing_ts_code():
    raw = _legacy_holdings(positions=[
        {"shares": 100, "mv": 1000, "industry": "银行"},
    ])
    result = normalize_account_state(raw)
    assert "positions[0].ts_code" in result["invalid_fields"]


def test_normalize_rejects_nonstandard_ts_code():
    raw = _legacy_holdings(positions=[
        {"ts_code": "not-a-code", "shares": 100, "mv": 1000, "industry": "银行"},
    ])
    result = normalize_account_state(raw)
    assert result["data_status"] == "incomplete"
    assert "positions[0].ts_code" in result["invalid_fields"]


def test_normalize_zero_shares_is_invalid():
    raw = _legacy_holdings(positions=[
        {"ts_code": "000001.SZ", "shares": 0, "mv": 0, "industry": "银行"},
    ])
    result = normalize_account_state(raw)
    assert any("shares" in f for f in result["invalid_fields"])


def test_normalize_unknown_does_not_explain_as_safe():
    # 未知额外字段不应导致报错,但也不被解释为0或安全
    raw = _legacy_holdings(positions=[
        {"ts_code": "000001.SZ", "shares": 100, "mv": 1000, "industry": "银行",
         "unknown_field": "some_value"},
    ])
    result = normalize_account_state(raw)
    assert result["data_status"] == "complete"
    assert result["positions"][0]["unknown_field"] == "some_value"


def test_normalize_non_dict_position():
    raw = _legacy_holdings(positions=["not a dict"])
    result = normalize_account_state(raw)
    assert "positions[0]" in result["invalid_fields"]


# ── freshness ──

def test_freshness_aligned():
    assert classify_account_freshness("20260807", "20260807") == FRESHNESS_ALIGNED
    assert classify_account_freshness("20260807", None) == FRESHNESS_ALIGNED


def test_freshness_missing():
    assert classify_account_freshness(None, "20260807") == FRESHNESS_MISSING


def test_freshness_invalid():
    assert classify_account_freshness("20260230", "20260807") == FRESHNESS_INVALID
    assert classify_account_freshness("not_a_date", "20260807") == FRESHNESS_INVALID
    assert classify_account_freshness("2026080", "20260807") == FRESHNESS_INVALID


def test_freshness_stale():
    assert classify_account_freshness("20260806", "20260807") == FRESHNESS_STALE
    assert classify_account_freshness("20250101", "20260807") == FRESHNESS_STALE


def test_freshness_future():
    assert classify_account_freshness("20260808", "20260807") == FRESHNESS_FUTURE
    assert classify_account_freshness("20300101", "20260807") == FRESHNESS_FUTURE


def test_require_account_as_of_aligned():
    account = {"as_of": "20260807"}
    require_account_as_of(account, "20260807")  # 不抛


def test_require_account_as_of_missing():
    with pytest.raises(AccountFreshnessError, match="ACCOUNT_FRESHNESS_ERROR"):
        require_account_as_of({"as_of": None}, "20260807")


def test_require_account_as_of_stale():
    with pytest.raises(AccountFreshnessError, match="ACCOUNT_AS_OF_STALE"):
        require_account_as_of({"as_of": "20260806"}, "20260807")


def test_require_account_as_of_future():
    with pytest.raises(AccountFreshnessError, match="ACCOUNT_AS_OF_FUTURE"):
        require_account_as_of({"as_of": "20260808"}, "20260807")


def test_require_account_as_of_invalid():
    with pytest.raises(AccountFreshnessError, match="ACCOUNT_AS_OF_INVALID"):
        require_account_as_of({"as_of": "20260230"}, "20260807")


# ── conditional_orders: legacy ──

def test_legacy_conditional_orders_string():
    raw = _legacy_holdings(conditional_orders="2张待核对")
    result = normalize_account_state(raw)
    co = result["conditional_orders"]
    assert co["status"] == "unverified"
    assert co["format"] == "legacy_free_text"
    assert co["verified_count"] is None
    assert "raw" not in co  # default: no raw


def test_legacy_conditional_orders_with_raw():
    raw = _legacy_holdings(conditional_orders="2张待核对")
    result = normalize_account_state(raw, include_raw_orders=True)
    co = result["conditional_orders"]
    assert co["raw"] == "2张待核对"


def test_legacy_conditional_orders_none():
    raw = _legacy_holdings(conditional_orders=None)
    result = normalize_account_state(raw)
    co = result["conditional_orders"]
    assert co["status"] == "missing"
    assert "raw" not in co


def test_legacy_conditional_orders_empty_string():
    raw = _legacy_holdings(conditional_orders="")
    result = normalize_account_state(raw)
    co = result["conditional_orders"]
    assert co["status"] == "missing"


# ── conditional_orders: v2 ──

def test_v2_orders_valid():
    v2 = _v2_order()
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "verified"
    assert result["verified_count"] == 1
    assert result["format"] == "structured_v2"


def test_v2_orders_valid_multiple():
    v2 = {
        "schema_version": 2,
        "orders": [
            {
                "order_id": "ord-001",
                "ts_code": "000001.SZ",
                "side": "BUY",
                "condition": {"field": "close", "operator": "<="},
                "price": 10.5,
                "shares": 100,
                "valid_from": "20260801",
                "valid_until": "20260831",
                "status": "active",
            },
            {
                "order_id": "ord-002",
                "ts_code": "000002.SZ",
                "side": "SELL",
                "condition": {"field": "close", "operator": ">="},
                "price": 20.0,
                "shares": 50,
                "valid_from": "20260801",
                "valid_until": "20260831",
                "status": "paused",
            },
        ],
    }
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "verified"
    assert result["verified_count"] == 2


def test_v2_orders_mixed_valid_invalid():
    v2 = {
        "schema_version": 2,
        "orders": [
            {
                "order_id": "ord-valid",
                "ts_code": "000001.SZ",
                "side": "BUY",
                "condition": {"field": "close", "operator": "<="},
                "price": 10.5,
                "shares": 100,
                "valid_from": "20260801",
                "valid_until": "20260831",
                "status": "active",
            },
            {
                "order_id": "ord-bad",
                "ts_code": "bad_code",
                "side": "BUY",
                "condition": {"field": "close", "operator": "<="},
                "price": -1.0,
                "shares": 0,
                "valid_from": "20260801",
                "valid_until": "20260831",
                "status": "active",
            },
        ],
    }
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert result["verified_count"] == 1
    assert len(result["invalid_fields"]) >= 3


def test_v2_orders_invalid_missing_order_id():
    v2 = _v2_order(orders=[{"ts_code": "000001.SZ", "side": "BUY",
                              "condition": {"field": "close", "operator": "<="},
                              "price": 10.5, "shares": 100,
                              "valid_from": "20260801", "valid_until": "20260831",
                              "status": "active"}])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert "order_id" in result["invalid_fields"][0]


def test_v2_orders_invalid_duplicate_order_id():
    v2 = _v2_order(orders=[
        {"order_id": "dup", "ts_code": "000001.SZ", "side": "BUY",
         "condition": {"field": "close", "operator": "<="},
         "price": 10.5, "shares": 100,
         "valid_from": "20260801", "valid_until": "20260831", "status": "active"},
        {"order_id": "dup", "ts_code": "000002.SZ", "side": "SELL",
         "condition": {"field": "close", "operator": ">="},
         "price": 20.0, "shares": 50,
         "valid_from": "20260801", "valid_until": "20260831", "status": "active"},
    ])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("重复" in f for f in result["invalid_fields"])


def test_v2_orders_invalid_ts_code():
    v2 = _v2_order(orders=[{**_v2_order()["orders"][0], "ts_code": "001218"}])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("ts_code" in f for f in result["invalid_fields"])


def test_v2_orders_invalid_side():
    v2 = _v2_order(orders=[{**_v2_order()["orders"][0], "side": "HOLD"}])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("side" in f for f in result["invalid_fields"])


def test_v2_orders_invalid_condition_field():
    v2 = _v2_order(orders=[
        {**_v2_order()["orders"][0], "condition": {"field": "open", "operator": "<="}},
    ])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("condition.field" in f for f in result["invalid_fields"])


def test_v2_orders_invalid_condition_operator():
    v2 = _v2_order(orders=[
        {**_v2_order()["orders"][0], "condition": {"field": "close", "operator": ">"}},
    ])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("condition.operator" in f for f in result["invalid_fields"])


def test_v2_orders_invalid_price():
    v2 = _v2_order(orders=[{**_v2_order()["orders"][0], "price": 0}])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("price" in f for f in result["invalid_fields"])

    v2_neg = _v2_order(orders=[{**_v2_order()["orders"][0], "price": -5}])
    result2 = validate_conditional_orders_v2(v2_neg)
    assert result2["status"] == "invalid"


def test_v2_orders_invalid_shares():
    v2 = _v2_order(orders=[{**_v2_order()["orders"][0], "shares": 100.5}])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("shares" in f for f in result["invalid_fields"])


@pytest.mark.parametrize(("field", "value"), [
    ("price", float("inf")),
    ("shares", float("inf")),
])
def test_v2_orders_reject_nonfinite_price_or_shares(field, value):
    order = {**_v2_order()["orders"][0], field: value}
    result = validate_conditional_orders_v2(_v2_order(orders=[order]))
    assert result["status"] == "invalid"
    assert any(field in item for item in result["invalid_fields"])


def test_v2_orders_invalid_date_order():
    v2 = _v2_order(orders=[
        {**_v2_order()["orders"][0], "valid_from": "20260831", "valid_until": "20260801"},
    ])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("valid_from" in f and "valid_until" in f for f in result["invalid_fields"])


def test_v2_orders_invalid_status():
    v2 = _v2_order(orders=[{**_v2_order()["orders"][0], "status": "triggered"}])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("status" in f for f in result["invalid_fields"])


def test_v2_orders_invalid_fake_date():
    v2 = _v2_order(orders=[
        {**_v2_order()["orders"][0], "valid_from": "20260230", "valid_until": "20261231"},
    ])
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "invalid"
    assert any("valid_from" in f for f in result["invalid_fields"])


def test_v2_orders_does_not_infer_trigger():
    # 价格条件不应被解释为已触发/已成交
    v2 = _v2_order()
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "verified"
    assert "triggered" not in str(result)


def test_v2_orders_unknown_schema_version():
    # schema_version=99 → unverified,不报verified
    v2 = {"schema_version": 99, "orders": []}
    result = validate_conditional_orders_v2(v2)
    assert result["status"] == "unverified"
    assert result["verified_count"] is None


def test_v2_orders_no_schema_version():
    result = validate_conditional_orders_v2({"orders": []})
    assert result["status"] == "unverified"


def test_structured_v2_include_raw_feeds_conditional_order_coverage():
    """normalize(include_raw=True) 必须把 orders 提到 coverage 能读的键上。

    MCP 默认 include_raw=False 仍不得暴露明细(NO_DETAIL,未知不解释为已覆盖)。
    """
    from ashare_gauntlet.stop_policy import conditional_order_coverage

    sell = {
        "order_id": "ord-sell",
        "ts_code": "000001.SZ",
        "side": "SELL",
        "condition": {"field": "close", "operator": "<="},
        "price": 10.5,
        "shares": 1000,
        "valid_from": "20260801",
        "valid_until": "20261231",
        "status": "active",
    }
    raw = _legacy_holdings(
        as_of="20260818",
        positions=[{
            "ts_code": "000001.SZ", "name": "平安银行", "industry": "银行",
            "shares": 1000, "cost": 12.5, "mv": 12500.0, "bucket": "长线",
        }],
        conditional_orders={"schema_version": 2, "orders": [sell]},
    )
    hidden = normalize_account_state(raw)
    assert hidden["conditional_orders"]["status"] == "verified"
    assert "orders" not in hidden["conditional_orders"]
    assert "raw" not in hidden["conditional_orders"]
    hidden_cov = conditional_order_coverage(hidden)
    assert hidden_cov["status"] == "NO_DETAIL"
    assert hidden_cov["uncovered"] == ["000001.SZ"]
    assert hidden_cov["covered"] == []

    shown = normalize_account_state(raw, include_raw_orders=True)
    assert shown["conditional_orders"]["orders"] == [sell]
    shown_cov = conditional_order_coverage(shown)
    assert shown_cov["status"] == "VERIFIED"
    assert shown_cov["covered"] == ["000001.SZ"]
    assert shown_cov["uncovered"] == []

    take_profit = {**sell, "condition": {"field": "close", "operator": ">="}}
    raw_tp = dict(raw, conditional_orders={"schema_version": 2, "orders": [take_profit]})
    tp_cov = conditional_order_coverage(
        normalize_account_state(raw_tp, include_raw_orders=True))
    assert tp_cov["status"] == "VERIFIED"
    assert tp_cov["covered"] == []
    assert tp_cov["uncovered"] == ["000001.SZ"]


# ── EOD valuation ──

def test_eod_valuation_shares_times_close():
    account = {
        "cash": 10000.0,
        "positions": [
            {"ts_code": "000001.SZ", "shares": 1000},
            {"ts_code": "000002.SZ", "shares": 500},
        ],
    }
    records = [
        {"ts_code": "000001.SZ", "close": 12.5, "error": None},
        {"ts_code": "000002.SZ", "close": 15.0, "error": None},
    ]
    val = build_eod_account_valuation(account, "20260807", records)
    assert val["status"] == "complete"
    assert val["basis"] == "shares_x_eod_close"
    assert val["market_value"] == 1000 * 12.5 + 500 * 15.0  # 12500 + 7500 = 20000
    assert val["total_assets"] == 20000 + 10000
    assert val["valued_position_count"] == 2


def test_eod_valuation_missing_close():
    account = {
        "cash": 10000.0,
        "positions": [
            {"ts_code": "000001.SZ", "shares": 1000},
            {"ts_code": "000002.SZ", "shares": 500},
        ],
    }
    records = [
        {"ts_code": "000001.SZ", "close": 12.5, "error": None},
        {"ts_code": "000002.SZ", "close": None, "error": "停牌"},
    ]
    val = build_eod_account_valuation(account, "20260807", records)
    assert val["status"] == "incomplete"
    assert val["market_value"] is None
    assert val["total_assets"] is None
    assert val["valued_position_count"] == 1


def test_eod_valuation_all_missing():
    account = {
        "cash": 10000.0,
        "positions": [
            {"ts_code": "000001.SZ", "shares": 1000},
        ],
    }
    records = [{"ts_code": "000001.SZ", "close": None, "error": "停牌"}]
    val = build_eod_account_valuation(account, "20260807", records)
    assert val["status"] == "incomplete"
    assert val["market_value"] is None
    assert val["valued_position_count"] == 0


def test_eod_valuation_no_positions_is_complete_cash_only_account():
    account = {"cash": 10000.0, "positions": []}
    val = build_eod_account_valuation(account, "20260807", [])
    assert val["status"] == "complete"
    assert val["market_value"] == 0.0
    assert val["total_assets"] == 10000.0
    assert val["position_count"] == 0


def test_eod_valuation_rejects_invalid_date_duplicate_records_and_nonfinite_close():
    account = {"cash": 10000.0, "positions": [
        {"ts_code": "000001.SZ", "shares": 100},
    ]}
    with pytest.raises(AccountSchemaError, match="真实日期"):
        build_eod_account_valuation(account, "20260230", [])

    duplicate = [
        {"ts_code": "000001.SZ", "close": 12.5, "error": None},
        {"ts_code": "000001.SZ", "close": 12.6, "error": None},
    ]
    assert build_eod_account_valuation(
        account, "20260807", duplicate)["status"] == "incomplete"
    nonfinite = [{"ts_code": "000001.SZ", "close": float("inf"), "error": None}]
    assert build_eod_account_valuation(
        account, "20260807", nonfinite)["status"] == "incomplete"


def test_eod_valuation_bad_shares():
    account = {
        "cash": 10000.0,
        "positions": [
            {"ts_code": "000001.SZ", "shares": 100.5},  # 非整数
        ],
    }
    records = [{"ts_code": "000001.SZ", "close": 12.5, "error": None}]
    val = build_eod_account_valuation(account, "20260807", records)
    assert val["status"] == "incomplete"
    assert val["market_value"] is None


def test_eod_valuation_does_not_use_manual_mv():
    # 即使 account 里有手工 mv,也不使用
    account = {
        "cash": 10000.0,
        "positions": [
            {"ts_code": "000001.SZ", "shares": 1000, "mv": 99999.0},
        ],
    }
    records = [{"ts_code": "000001.SZ", "close": 12.5, "error": None}]
    val = build_eod_account_valuation(account, "20260807", records)
    assert val["market_value"] == 12500.0  # shares * close, not 99999


def test_normalize_with_expected_as_of():
    raw = _legacy_holdings()
    result = normalize_account_state(raw, expected_as_of="20260807")
    assert result["freshness"] == FRESHNESS_ALIGNED


def test_normalize_with_stale_as_of():
    raw = _legacy_holdings()
    result = normalize_account_state(raw, expected_as_of="20260808")
    assert result["freshness"] == FRESHNESS_STALE


def test_normalize_invalid_as_of_preserves_invalid_freshness():
    result = normalize_account_state(
        _legacy_holdings(as_of="20260230"), expected_as_of="20260807")
    assert result["as_of"] is None
    assert result["freshness"] == FRESHNESS_INVALID
    assert "as_of" in result["invalid_fields"]
    with pytest.raises(AccountFreshnessError, match="ACCOUNT_AS_OF_INVALID"):
        require_account_as_of(result, "20260807")


@pytest.mark.parametrize(("field", "value"), [
    ("cash", float("inf")),
    ("cash", float("-inf")),
])
def test_normalize_rejects_nonfinite_cash(field, value):
    result = normalize_account_state(_legacy_holdings(**{field: value}))
    assert result["data_status"] == "incomplete"
    assert field in result["invalid_fields"]


@pytest.mark.parametrize("field", ["shares", "cost", "mv"])
def test_normalize_rejects_nonfinite_position_values(field):
    raw = _legacy_holdings()
    raw["positions"][0][field] = float("inf")
    result = normalize_account_state(raw)
    assert result["data_status"] == "incomplete"
    assert any(item.endswith(f".{field}") for item in result["invalid_fields"])


def test_eod_valuation_infinite_shares_is_incomplete_not_overflow():
    account = {"cash": 10000.0, "positions": [
        {"ts_code": "000001.SZ", "shares": float("inf")},
    ]}
    records = [{"ts_code": "000001.SZ", "close": 12.5, "error": None}]
    result = build_eod_account_valuation(account, "20260807", records)
    assert result["status"] == "incomplete"
    assert result["market_value"] is None


def test_import_has_no_side_effects():
    """在隔离进程检查 import/reload，避免污染主 pytest 的异常类 identity。"""
    root = Path(__file__).resolve().parents[1]
    code = (
        "import importlib; "
        "import ashare_gauntlet.account_state as acs; "
        "importlib.reload(acs)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=root,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
