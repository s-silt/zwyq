"""X-01 IVOL 窗口稳健性:--ivol-windows 解析纯函数(experiments.md 预注册 {21,63,252})。"""
from __future__ import annotations

import pytest


def test_parse_ivol_windows_happy_path():
    from scripts.factor_backtest import parse_ivol_windows

    assert parse_ivol_windows(None) == []
    assert parse_ivol_windows("") == []
    assert parse_ivol_windows("21,63,252") == [21, 63, 252]


def test_parse_ivol_windows_fails_loud():
    from scripts.factor_backtest import parse_ivol_windows

    with pytest.raises(ValueError):
        parse_ivol_windows("21,abc")        # 非整数
    with pytest.raises(ValueError):
        parse_ivol_windows("21,-5")         # 非正
    with pytest.raises(ValueError):
        parse_ivol_windows("21,21")         # 重复窗口=同名因子列互相覆盖
