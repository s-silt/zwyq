"""Tests for aggressive_pick —— 进攻性视图纯函数(质量地板 + 扣非增速排序 + 硬排除)。

只测纯函数层(quality_floor / aggressive_rank),不打 API 不读缓存:
- 质量地板:decile==10 且 tier=='🟢' 之外全部剔除(地板不动);
- 硬排除:排除行业被剔、已持仓被剔;
- 排序:按扣非增速横截面百分位降序(percentile_rank 既有约定,最大=1);
- 缺失:增速缺失/NaN 排最后、字段如实 None,不伪造 0/中位数;
- 空池:返回空列表不崩;
- 输入不被就地修改(纯函数边界)。
"""
import math

import pandas as pd

from scripts.aggressive_pick import EXCLUDED_INDUSTRIES, aggressive_rank, quality_floor


def _row(code: str, industry: str = "小金属", **kw) -> dict:
    base = {"ts_code": code, "name": f"N{code}", "industry": industry,
            "decile": 10, "tier": "🟢"}
    base.update(kw)
    return base


# ---------- 质量地板:D10 + 🟢(既有约定,进攻不降质量) ----------

def test_quality_floor_keeps_only_d10_green():
    rows = [_row("A"), _row("B", decile=9), _row("C", tier="🟡"),
            _row("D", decile=None, tier="🟢"), _row("E")]
    kept = quality_floor(rows)
    assert [r["ts_code"] for r in kept] == ["A", "E"]


# ---------- 硬排除:行业 / 已持仓 ----------

def test_excluded_industry_removed():
    rows = [_row("A", industry="通信设备"), _row("B", industry="小金属"),
            _row("C", industry="半导体")]
    out = aggressive_rank(rows, {"A": 50.0, "B": 10.0, "C": 99.0},
                          {"通信设备", "半导体"}, set())
    assert [r["ts_code"] for r in out] == ["B"]


def test_held_codes_removed():
    rows = [_row("A"), _row("B")]
    out = aggressive_rank(rows, {"A": 50.0, "B": 10.0}, set(), {"A"})
    assert [r["ts_code"] for r in out] == ["B"]


# ---------- 排序:扣非增速横截面百分位降序 ----------

def test_sorted_by_growth_percentile_desc():
    rows = [_row("A"), _row("B"), _row("C")]
    out = aggressive_rank(rows, {"A": 10.0, "B": 30.0, "C": 20.0}, set(), set())
    assert [r["ts_code"] for r in out] == ["B", "C", "A"]
    # 百分位 = 池内横截面 rank(pct=True) 既有约定(factor_model.percentile_rank,最大=1)
    assert math.isclose(out[0]["growth_pct"], 1.0)
    assert math.isclose(out[1]["growth_pct"], 2 / 3)
    assert math.isclose(out[2]["growth_pct"], 1 / 3)
    # dt_yoy 原值透传(展示用),排序不改数
    assert out[0]["dt_yoy"] == 30.0
    # 纯函数边界:入参行未被就地污染
    assert "growth_pct" not in rows[0] and "dt_yoy" not in rows[0]


def test_percentile_computed_within_post_exclusion_pool():
    # 横截面 = 硬排除后的候选池:被剔除的高增速票不应占用分位名额
    rows = [_row("X", industry="半导体"), _row("A"), _row("B")]
    out = aggressive_rank(rows, {"X": 999.0, "A": 20.0, "B": 10.0}, {"半导体"}, set())
    assert [r["ts_code"] for r in out] == ["A", "B"]
    assert math.isclose(out[0]["growth_pct"], 1.0)   # 池内最大=1,不因 X 存在而变 2/3


# ---------- 缺失:排最后、不伪造 ----------

def test_missing_growth_sorts_last_not_fabricated():
    rows = [_row("A"), _row("B"), _row("C")]
    # B 完全缺失,C 是 NaN —— 都视为缺失;A 增速为负也照排(负增长≠缺失)
    out = aggressive_rank(rows, {"A": -5.0, "C": float("nan")}, set(), set())
    assert out[0]["ts_code"] == "A"
    assert [r["ts_code"] for r in out[1:]] == ["B", "C"]   # 缺失按原池序稳定排最后
    for r in out[1:]:
        assert r["dt_yoy"] is None and r["growth_pct"] is None  # 如实 None,不伪造 0/中位数


# ---------- 空池 ----------

def test_empty_pool_returns_empty():
    assert aggressive_rank([], {}, set(), set()) == []
    # 全被硬排除同样返回空,不崩
    rows = [_row("A", industry="元器件"), _row("B")]
    assert aggressive_rank(rows, {}, {"元器件"}, {"B"}) == []


# ---------- 排除集合口径钉死(基金变了要同步改,测试提醒) ----------

def test_excluded_industries_matches_fund_topten_derivation():
    # 出处:用户场外基金(华泰柏瑞质量精选/质量成长、易方达远见成长)前十大重仓
    # 所属 tushare 行业推导 —— 基金持仓变了必须同步改 data/profile.json 与本测试
    assert EXCLUDED_INDUSTRIES == {"通信设备", "元器件", "半导体", "IT设备"}


# ---------- 个人 profile 配置层(外部评审 P2:个人约束出代码入配置) ----------

def test_load_excluded_industries_reads_profile(tmp_path):
    import json

    from scripts.aggressive_pick import load_excluded_industries

    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"excluded_industries": ["银行", "白酒"]}), encoding="utf-8")
    assert load_excluded_industries(str(p)) == {"银行", "白酒"}


def test_load_excluded_industries_fails_loud(tmp_path):
    # 排除约束静默失效 = 风险敞口翻倍,必须 fail-loud:文件缺失/键缺失都要炸
    import json

    import pytest

    from scripts.aggressive_pick import load_excluded_industries

    with pytest.raises(FileNotFoundError):
        load_excluded_industries(str(tmp_path / "nope.json"))
    p = tmp_path / "profile.json"
    p.write_text(json.dumps({"其他键": []}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_excluded_industries(str(p))


# ---------- latest_rows PIT(as_of):未来公告不得混入横截面 ----------
# 场景是真实风险:价格缓存停在旧日、财务缓存已刷到新公告 → 不带 as_of 会拿"未来财报"
# 排今天的横截面(前视偏差)。aggressive_pick 与 factor_rank 同用 latest_rows,在此钉口径。

def _fina_cache(tmp_path) -> None:
    d = tmp_path / "fina_indicator"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"ts_code": "600000.SH", "end_date": "20250331", "ann_date": "20250428",
         "update_flag": "0", "dt_netprofit_yoy": 10.0, "tr_yoy": 5.0},
        {"ts_code": "600000.SH", "end_date": "20250630", "ann_date": "20250815",
         "update_flag": "0", "dt_netprofit_yoy": 99.0, "tr_yoy": 50.0},
    ]).to_parquet(d / "600000.SH.parquet", index=False)


def test_latest_rows_pit_excludes_future_announcements(tmp_path, monkeypatch):
    import scripts.factor_rank as fr
    _fina_cache(tmp_path)
    monkeypatch.setattr(fr, "CACHE", str(tmp_path))
    # 价格缓存停在 20250701:Q2 财报 8/15 才公告,是未来信息 → 必须取 Q1
    out = fr.latest_rows("fina_indicator", ["dt_netprofit_yoy", "tr_yoy"], as_of="20250701")
    assert str(out.loc["600000.SH", "end_date"]) == "20250331"
    assert float(out.loc["600000.SH", "dt_netprofit_yoy"]) == 10.0


def test_latest_rows_pit_includes_rows_announced_on_or_before_as_of(tmp_path, monkeypatch):
    import scripts.factor_rank as fr
    _fina_cache(tmp_path)
    monkeypatch.setattr(fr, "CACHE", str(tmp_path))
    # as_of 恰为 Q2 公告日(<= 含当日)→ 正常取 Q2 最新报告期
    out = fr.latest_rows("fina_indicator", ["dt_netprofit_yoy", "tr_yoy"], as_of="20250815")
    assert str(out.loc["600000.SH", "end_date"]) == "20250630"
    assert float(out.loc["600000.SH", "dt_netprofit_yoy"]) == 99.0


def test_latest_rows_pit_code_with_no_announced_rows_excluded(tmp_path, monkeypatch):
    import scripts.factor_rank as fr
    _fina_cache(tmp_path)
    d = tmp_path / "fina_indicator"
    pd.DataFrame([
        {"ts_code": "600001.SH", "end_date": "20250630", "ann_date": "20250815",
         "update_flag": "0", "dt_netprofit_yoy": 1.0, "tr_yoy": 1.0},
    ]).to_parquet(d / "600001.SH.parquet", index=False)
    monkeypatch.setattr(fr, "CACHE", str(tmp_path))
    # 某票在 as_of 时点尚无任何已公告行 → 如实不入横截面,不拿未来行凑数
    out = fr.latest_rows("fina_indicator", ["dt_netprofit_yoy", "tr_yoy"], as_of="20250401")
    assert "600001.SH" not in out.index
    assert "600000.SH" not in out.index   # 600000 的 Q1 也是 4/28 才公告
