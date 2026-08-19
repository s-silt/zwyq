"""TDD:C 层 IO 编排 scripts/dashboard.py 的容错层。

铁律(见 memory analysis-priorities):纯渲染层(render_html/render_svg)保持 fail-loud、
不吞错;容错放 IO 层——坏 record 被「跳过 + 响亮 surface(报 ts_code)」,既不污染
dashboard、又不掩盖数据不纯,dashboard 仍可用。这里只验证 IO 层的 partition/上报逻辑。
"""
from __future__ import annotations

import scripts.dashboard as d


def test_partition_renderable_skips_and_reports_bad(monkeypatch) -> None:
    good_rec = {"ts_code": "600000.SH"}
    bad_rec = {"ts_code": "000001.SZ"}

    def fake_svg(rec, cohort=None):
        if rec.get("ts_code") == "000001.SZ":
            raise KeyError("坏 record:缺字段")
        return "<svg></svg>"

    monkeypatch.setattr(d, "render_svg_card", fake_svg)
    good, failures = d._partition_renderable([good_rec, bad_rec])

    # 坏 record 被剔除、好 record 保留
    assert good == [good_rec]
    # 失败被 surface:带 ts_code + 错误类型,不静默
    assert len(failures) == 1
    assert failures[0][0] == "000001.SZ"
    assert "KeyError" in failures[0][1]


def test_partition_renderable_all_good_no_failures(monkeypatch) -> None:
    recs = [{"ts_code": "600000.SH"}, {"ts_code": "600519.SH"}]
    monkeypatch.setattr(d, "render_svg_card", lambda rec, cohort=None: "<svg></svg>")
    good, failures = d._partition_renderable(recs)
    assert good == recs
    assert failures == []


def test_html_console_retired_cli_fails_loud() -> None:
    """退役后 CLI 必须明确失败并指向 q today——静默产出旧页会让陈旧页面看起来像当前决策。

    (跨层审计 Q3 + 用户 2026-08-19 拍板;历史页面只读留档并带废弃水印)
    """
    import pytest

    with pytest.raises(SystemExit, match="已退役"):
        d.main()
    with pytest.raises(SystemExit, match="daily_brief"):
        d.main("20260618")


def test_cards_no_longer_emits_html() -> None:
    """cards 的 render_outputs 只出 md;HTML 分支退役,不得再产出可被误读为决策的页面。"""
    import inspect

    from scripts import cards

    src = inspect.getsource(cards.render_outputs)
    assert "render_dashboard" not in src
    assert '"html"' not in src
