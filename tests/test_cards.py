"""TDD:D 编排层 scripts/cards.py 的纯粹可测部分 —— 落库 JSON 的 NaN 兜底防线(#1)。

不跑 main()(联网/读缓存);只测 dump_cards 的序列化守卫:任何漏网的 NaN
必须响亮抛 ValueError,而非把非法 JSON 字面量 ``NaN`` 写进库(数据源纯净)。
"""
import json

import pytest

from scripts.cards import dump_cards


def test_dump_cards_writes_clean_record(tmp_path):
    out = tmp_path / "20260618.json"
    records = [{"ts_code": "601138.SH", "valuation": {"pe_ttm": 20.0, "peg": None}}]
    dump_cards(records, str(out))
    # 回读:None 正常落为 null;数值原样
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded[0]["valuation"]["pe_ttm"] == 20.0
    assert loaded[0]["valuation"]["peg"] is None


def test_dump_cards_rejects_nan_loudly(tmp_path):
    # allow_nan=False:漏到落库层的 NaN 抛 ValueError,不写非标准 JSON 字面量 NaN
    out = tmp_path / "20260618.json"
    records = [{"ts_code": "601138.SH", "technical": {"ret20": float("nan")}}]
    with pytest.raises(ValueError):
        dump_cards(records, str(out))


def test_dump_cards_rejects_infinity_loudly(tmp_path):
    out = tmp_path / "20260618.json"
    records = [{"ts_code": "601138.SH", "technical": {"vol_ratio": float("inf")}}]
    with pytest.raises(ValueError):
        dump_cards(records, str(out))
