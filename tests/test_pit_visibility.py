"""重述前视修复(四镜头验证 P1):更正行可见日=f_ann_date,非原 ann_date。

实证背景:income/cashflow/balancesheet 的重述行(update_flag=1)保留**原公告日**
ann_date,真实发布日在 f_ann_date(实测重述滞后中位 ~360 天,max 4001 天)——
_pit 按 ann_date 过滤会让更正值在原公告日即"可见",构成前视。
"""
from __future__ import annotations

import pandas as pd


def _rows() -> pd.DataFrame:
    # 000003.SZ 实测形态:同 (end_date, ann_date) 两行,f_ann_date 相差两年
    return pd.DataFrame({
        "ts_code": ["A", "A"],
        "end_date": ["20201231", "20201231"],
        "ann_date": ["20210501", "20210501"],
        "f_ann_date": ["20210501", "20230429"],
        "update_flag": ["0", "1"],
        "v": [-687904.0, -1287904.0],
    })


def test_restated_visibility_coalesces_f_ann_date():
    from ashare_gauntlet.backtest import restated_visibility

    out = restated_visibility(_rows())
    assert "f_ann_date" not in out.columns
    assert list(out["ann_date"]) == ["20210501", "20230429"]   # 更正行可见日=真实发布日
    # 无 f_ann_date 列的表(fina_indicator)原样返回,不报错
    plain = _rows().drop(columns=["f_ann_date"])
    assert restated_visibility(plain) is plain


def test_pit_does_not_see_restatement_before_release():
    from ashare_gauntlet.backtest import restated_visibility
    from scripts.factor_backtest import _pit

    fixed = restated_visibility(_rows()).sort_values(
        ["ts_code", "end_date", "ann_date", "update_flag"], kind="mergesort")
    assert _pit(fixed, "20220101")["v"]["A"] == -687904.0    # 更正发布前:只见原值
    assert _pit(fixed, "20230601")["v"]["A"] == -1287904.0   # 发布后:取更正值


def test_restated_visibility_handles_numeric_f_ann_date():
    from ashare_gauntlet.backtest import restated_visibility

    # Codex P2:parquet 可能把日期存成 float(20230429.0)——astype(str) 会得
    # "20230429.0" 被八位正则静默拒绝,修复失效;须先规范为整数日期字符串
    df = _rows()
    df["f_ann_date"] = [20210501.0, 20230429.0]
    out = restated_visibility(df)
    assert list(out["ann_date"]) == ["20210501", "20230429"]


def test_restated_visibility_ignores_bad_f_ann_date():
    from ashare_gauntlet.backtest import restated_visibility

    df = _rows()
    df.loc[1, "f_ann_date"] = None                            # 缺失→保留原 ann_date
    out = restated_visibility(df)
    assert list(out["ann_date"]) == ["20210501", "20210501"]
