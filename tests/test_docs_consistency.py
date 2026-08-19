"""同一统计量的跨文件副本一致性。

这些数字是"某条规则为什么这样定"的唯一依据:DP 被否决时引用的 ACC 对照、composite
组件"各自过线有余量"的证据、C2 优于立即退出的幅度。同一事实出现两个值时,复算者
无法判断哪份是修正后口径——X-05/X-06 复跑各漏更过一份副本,X-10 结案又新抄错一份。
所以这里**不钉死数值**:从权威表里解析当期值再断言引用处与之相等。权威读数以后再改,
漏更的副本会自己变红,而不需要有人记得回来改测试。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    """读文件;**不入公开仓的路径缺失时 skip 而非 fail**。

    `/data/`、`/docs/`、`/README.md` 按 2026-08-12 的决定不进版本控制(含个人账户状态
    与研究档案),所以干净 clone 上这些文件不存在。此处 skip 让公开仓不假红,而本地
    (文件都在)照常守住跨文件数字漂移——守卫的价值本就在本地维护现场。
    """
    # 在函数内读:import 阶段不做文件 I/O(项目约定)
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} 不在版本控制内(见 .gitignore),跳过跨文件一致性校验")
    return path.read_text(encoding="utf-8")


def _section5_nw_t(methodology: str, factor: str) -> str:
    """§5 现役六因子表里该因子的 NW t(权威主产物口径,去掉正号)。"""
    m = re.search(rf"^\|\s*{factor}\s*\|[^|]*\|\s*\*\*([^*|]+?)\*\*\s*\|", methodology, re.M)
    assert m, f"methodology §5 表中找不到 {factor} 行"
    return m.group(1).strip().lstrip("+")


def _increment_nw_t(methodology: str, factor: str) -> str:
    """§10 增量准入表(X-02/X-03)里该因子的增量 NW t。"""
    table = methodology.split("增量准入实验", 1)
    assert len(table) == 2, "methodology 中找不到增量准入实验小节"
    m = re.search(rf"^\|\s*{factor}\s*\|[^|]*\|\s*\+?([0-9.]+)\s*\|", table[1], re.M)
    assert m, f"methodology §10 增量表中找不到 {factor} 行"
    return m.group(1)


def _m3_net(methodology: str, variant_pattern: str) -> str:
    """§10 M3 退出规则实验里某变体的净超额/期(X-05 修正后口径)。"""
    m = re.search(rf"{variant_pattern}:净 \+([0-9.]+)%", methodology)
    assert m, f"methodology §10 M3 中找不到 {variant_pattern}"
    return m.group(1)


def test_methodology_inline_ep_t_matches_section5():
    # §10 判读2 自称引用"§5 主产物口径",两处必须逐位相同(曾停在 X-06 前的 6.62)
    methodology = _read("docs/methodology.md")
    assert f"EP t{_section5_nw_t(methodology, 'EP')}" in methodology


def test_x10_cites_current_acc_increment_t():
    # X-10 拿 ACC 增量 t 作 DP 否决的形态对照,须引 X-02 的定谳值
    acc_t = _increment_nw_t(_read("docs/methodology.md"), "ACC")
    assert f"同 ACC(X-02 t{acc_t})形态" in _read("docs/experiments.md")


def test_portfolio_decision_comment_matches_m3():
    # 生产注释里的 C2 幅度须是 X-05 修正后口径,不能停在 M3 首跑读数
    methodology = _read("docs/methodology.md")
    c2 = _m3_net(methodology, r"连续 2 期确认\(C2\)")
    immediate = _m3_net(methodology, r"立即退出\(现行 PROD\)")
    src = _read("ashare_gauntlet/portfolio_decision.py")
    assert f"净+{c2}% vs 立即退出+{immediate}%" in src


def test_readme_does_not_pin_a_test_count():
    # 写死的测试项数每加一个测试就旧一次,只留命令不留数字
    assert re.search(r"当前\s*[\d,]+\s*项", _read("README.md")) is None
