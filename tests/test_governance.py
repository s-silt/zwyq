"""治理雷结构化(pledge_detail / fina_audit)纯函数测试。

口径以实测为准(2026-07-02 对 002602.SZ / 600989.SH 探字段):
- pledge_detail 解押**不回写**旧行 is_release,而是另生成 is_release==1 新行,
  对 is_release==0 求和会重复计数(实测王佶 sum≈267530 万股 vs 真实快照 75463),
  部分解押还会漏计(实测永丰国际 sum=0 但快照 2500)。唯一正确口径 =
  每股东最新 ann_date 行的 pledged_amount(未解押总量快照,全解押后实测归 0)。
- h_total_ratio 实测 = 持股总数占总股本比(%),**不是**质押占其持股比;
  质押占其持股比 = pledged_amount / holding_amount(同一公告快照的定义性比值)。
"""

import pandas as pd
import pytest

from ashare_gauntlet.governance import audit_opinion, controller_pledge


def _pledge_row(**kw) -> dict:
    """一行 pledge_detail 夹具,列名与实测返回一致。"""
    base = {
        "ts_code": "002602.SZ", "ann_date": "20251220", "holder_name": "王佶",
        "pledge_amount": 1800.0, "start_date": "20251218", "end_date": None,
        "is_release": "0", "release_date": None, "pledgor": "某银行",
        "holding_amount": 76404.56, "pledged_amount": 75462.78,
        "p_total_ratio": 0.24, "h_total_ratio": 10.36, "is_buyback": "0",
    }
    base.update(kw)
    return base


# ---------- controller_pledge ----------

def test_controller_pledge_empty_returns_empty_dict():
    assert controller_pledge(pd.DataFrame()) == {}
    assert controller_pledge(None) == {}


def test_controller_pledge_uses_latest_snapshot_not_is_release_sum():
    # 上海吉运盛:旧质押行 is_release 仍为 0,但最新公告快照 pledged_amount=0(已全解押)
    # —— 若错误地对 is_release==0 求和,会把 16205.71 万股当成未解押。
    df = pd.DataFrame([
        _pledge_row(holder_name="上海吉运盛", ann_date="20260622", pledge_amount=16205.71,
                    is_release="0", holding_amount=19605.71, pledged_amount=16205.71,
                    p_total_ratio=2.20, h_total_ratio=2.66),
        _pledge_row(holder_name="上海吉运盛", ann_date="20260701", pledge_amount=10844.30,
                    is_release="1", release_date="20260629", holding_amount=14346.10,
                    pledged_amount=0.0, p_total_ratio=1.47, h_total_ratio=1.95),
        _pledge_row(),  # 王佶 20251220 快照:75462.78/76404.56 未解押
    ])
    top = controller_pledge(df)
    assert top["holder_name"] == "王佶"
    assert top["pledged_ratio_of_holding"] == pytest.approx(98.767, abs=0.01)
    # 占总股本 = 占持股比 × 持股占总股本比(h_total_ratio)
    assert top["ratio_of_total"] == pytest.approx(98.767 * 10.36 / 100, abs=0.01)
    assert top["asof"] == "20251220"


def test_controller_pledge_picks_max_ratio_holder():
    df = pd.DataFrame([
        _pledge_row(holder_name="低比例大户", holding_amount=100000.0, pledged_amount=30000.0),
        _pledge_row(holder_name="高比例小户", holding_amount=5000.0, pledged_amount=4500.0),
    ])
    top = controller_pledge(df)
    assert top["holder_name"] == "高比例小户"
    assert top["pledged_ratio_of_holding"] == pytest.approx(90.0)


def test_controller_pledge_ratio_tie_breaks_by_pledged_amount():
    # 实测 002602 有多个 100% 质押股东 —— 同比例时质押量大者风险敞口大。
    df = pd.DataFrame([
        _pledge_row(holder_name="小100%", holding_amount=4000.0, pledged_amount=4000.0),
        _pledge_row(holder_name="大100%", holding_amount=47581.39, pledged_amount=47581.39),
    ])
    assert controller_pledge(df)["holder_name"] == "大100%"


def test_controller_pledge_missing_h_total_ratio_keeps_holding_ratio():
    # h_total_ratio 缺 → 占总股本比 None 不伪造;占其持股比仍可算(同行两字段都在)。
    df = pd.DataFrame([_pledge_row(h_total_ratio=float("nan"))])
    top = controller_pledge(df)
    assert top["pledged_ratio_of_holding"] == pytest.approx(98.767, abs=0.01)
    assert top["ratio_of_total"] is None


def test_controller_pledge_missing_holding_amount_gives_none_not_fabricated():
    # holding_amount 缺 → 占其持股比 None;该股东仍被上报(未知≠无风险),不静默丢弃。
    df = pd.DataFrame([_pledge_row(holding_amount=float("nan"))])
    top = controller_pledge(df)
    assert top["holder_name"] == "王佶"
    assert top["pledged_ratio_of_holding"] is None
    assert top["ratio_of_total"] is None


def test_controller_pledge_prefers_computable_ratio_over_unknown():
    # 可算比例的股东优先于比例未知的股东(未知者不参与"最高比例"排序,但兜底可返回)。
    df = pd.DataFrame([
        _pledge_row(holder_name="未知比例", holding_amount=float("nan"), pledged_amount=99999.0),
        _pledge_row(holder_name="已知50%", holding_amount=10000.0, pledged_amount=5000.0),
    ])
    assert controller_pledge(df)["holder_name"] == "已知50%"


def test_controller_pledge_all_released_returns_empty():
    # 全部股东最新快照 pledged_amount=0 → 当前无未解押质押。
    df = pd.DataFrame([
        _pledge_row(holder_name="甲", pledged_amount=0.0, is_release="1"),
        _pledge_row(holder_name="乙", pledged_amount=0.0, is_release="1"),
    ])
    assert controller_pledge(df) == {}


def test_controller_pledge_int_ann_date_from_parquet_roundtrip():
    # parquet 落盘再读回时 ann_date 可能变成 int64 —— 口径不受 dtype 影响。
    df = pd.DataFrame([
        _pledge_row(ann_date=20251220),
        _pledge_row(ann_date=20200101, pledged_amount=0.0, is_release="1"),
    ])
    top = controller_pledge(df)
    assert top["asof"] == "20251220"
    assert top["pledged_ratio_of_holding"] == pytest.approx(98.767, abs=0.01)


# ---------- audit_opinion ----------

def _audit_row(**kw) -> dict:
    base = {
        "ts_code": "600989.SH", "ann_date": "20260313", "end_date": "20251231",
        "audit_result": "标准无保留意见", "audit_fees": 4200000.0,
        "audit_agency": "安永华明会计师事务所", "audit_sign": "孙芳,刘小红",
    }
    base.update(kw)
    return base


def test_audit_opinion_empty_returns_empty_dict():
    assert audit_opinion(pd.DataFrame()) == {}
    assert audit_opinion(None) == {}


def test_audit_opinion_latest_period_standard():
    df = pd.DataFrame([
        _audit_row(ann_date="20250312", end_date="20241231"),
        _audit_row(),  # 最新期 20251231
    ])
    op = audit_opinion(df)
    assert op["end_date"] == "20251231"
    assert op["audit_result"] == "标准无保留意见"
    assert op["is_nonstandard"] is False


@pytest.mark.parametrize("result", [
    "保留意见",
    "无法表示意见",
    "带强调事项段的无保留意见",  # 含"无保留"但不含"标准无保留" —— 仍是红旗
    "否定意见",
])
def test_audit_opinion_nonstandard_flags(result: str):
    op = audit_opinion(pd.DataFrame([_audit_row(audit_result=result)]))
    assert op["is_nonstandard"] is True
    assert op["audit_result"] == result


def test_audit_opinion_missing_result_is_nonstandard_true():
    # audit_result 缺失 → 不能当"标准"吞掉,按非标上报(surface,人工去查)。
    op = audit_opinion(pd.DataFrame([_audit_row(audit_result=None)]))
    assert op["is_nonstandard"] is True
    assert op["audit_result"] is None


def test_audit_opinion_same_period_takes_latest_announcement():
    # 同一报告期两次公告(更正)→ 以 ann_date 最新为准。
    df = pd.DataFrame([
        _audit_row(ann_date="20260313", audit_result="保留意见"),
        _audit_row(ann_date="20260420", audit_result="标准无保留意见"),
    ])
    op = audit_opinion(df)
    assert op["audit_result"] == "标准无保留意见"
    assert op["is_nonstandard"] is False
