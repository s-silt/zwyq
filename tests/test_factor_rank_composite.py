"""composite 因子集测试 —— 钉住"哪些因子入分"这一模型定义本身。

演进(权威读数=docs/methodology.md §5/§6/§10 增量表,代码与测试只复述不自带版本):
- 2026-07-05:5→3(EP+BP+ACC),剔 ROE/GP(12年噪声+与 EP/BP 冗余~0.6);
- 2026-07-06 上午:3→2(EP+BP),依据是当时的 ACC 读数(t 2.36/真实净≈0);
- 2026-07-06 晚:2→3(**EP+BP+IVOL负向**),IVOL 过完整门禁;
- **2026-07-10 数据修复(70 个月末 total_mv 空文件回填)推翻上面两条的读数依据**:
  ACC 复跑后 t+4.05 五门全过,不入分的真实理由改为 X-02 common-support 增量 t+1.63
  未过门;MAX/NLIMIT 同理(X-03 增量 t0.73/0.74、MAX-IVOL 冗余 0.78)。
新因子入 composite 须先过五门(NW t>3+真实净>0+LOYO+市场状态+**多头腿成本后>0**),
再过增量门(对现役 composite 的 common-support 增量 |NW t|>3)——两道门缺一不可。
"""
from pathlib import Path

import pandas as pd

from scripts.factor_rank import COMPOSITE_FACTORS, composite_inputs_complete, composite_score


def test_composite_factors_are_the_validated_three():
    # 入分=EP+BP+IVOL(f_IVOL 已是"低波=高分"方向的百分位);其余全展示列
    assert set(COMPOSITE_FACTORS) == {"f_EP", "f_BP", "f_IVOL"}


def test_composite_score_invariant_to_display_factors():
    # 行为测试:扰动展示列不改变 score
    base = pd.DataFrame({
        "f_EP": [0.9, 0.1], "f_BP": [0.6, 0.4], "f_IVOL": [0.3, 0.7],
        "f_ACC": [0.99, 0.01], "f_ROE": [0.99, 0.01], "f_GPOA": [0.99, 0.01], "f_MOM": [0.99, 0.01],
    })
    perturbed = base.copy()
    display = ["f_ACC", "f_ROE", "f_GPOA", "f_MOM"]
    perturbed[display] = 1.0 - perturbed[display]
    pd.testing.assert_series_equal(composite_score(base), composite_score(perturbed))
    assert abs(composite_score(base).iloc[0] - (0.9 + 0.6 + 0.3) / 3) < 1e-12


def test_composite_score_skips_missing_factor():
    df = pd.DataFrame({"f_EP": [0.8], "f_BP": [None], "f_IVOL": [0.4]})
    assert abs(composite_score(df).iloc[0] - 0.6) < 1e-12


def test_composite_inputs_complete_gate_three_factor():
    # 入池门槛=EP/BP/IVOL 原值全齐(新上市<21日无 IVOL → 如实清退);展示列缺失不清退
    df = pd.DataFrame({
        "EP": [0.10, 0.10, None], "BP": [0.50, 0.50, 0.50], "IVOL": [0.02, None, 0.02],
        "ACC": [None, 0.02, 0.02], "roe": [None, 9.9, 9.9], "GPOA": [None, 0.3, 0.3]})
    assert composite_inputs_complete(df).tolist() == [True, False, False]


def test_spec_crowd_flags_union_of_family_top_decile():
    # 🎰投机拥挤标签:IVOL/MAX/NLIMIT 任一处于当日横截面 top decile(to_decile==10,
    # 复用十分位既有约定,零新常数;union 为定义性聚合)。原值口径(非中性化)——
    # 标签回答"这票现在是不是彩票票",人类可读性优先
    from scripts.factor_rank import spec_crowd_flags
    n = 20
    ivol = pd.Series([0.01] * n, index=[f"c{i}" for i in range(n)])
    mx = pd.Series([0.02] * n, index=ivol.index)
    nl = pd.Series([0.0] * n, index=ivol.index)
    ivol.iloc[0] = 0.9        # c0 仅 IVOL 顶
    mx.iloc[1] = 0.5          # c1 仅 MAX 顶
    nl.iloc[2] = 8.0          # c2 仅 NLIMIT 顶
    flags = spec_crowd_flags(ivol, mx, nl)
    assert bool(flags["c0"]) and bool(flags["c1"]) and bool(flags["c2"])
    assert not bool(flags["c5"])

# ---- 读数漂移守卫 ----------------------------------------------------------
# 为什么要机器核对而不是靠人盯:2026-07-10 修复 70 个月末 total_mv 空文件后权威 t
# 全线上移(EP 4.36→6.59、ACC 2.36→4.05),但注释里的那份读数副本没跟着改,
# "ACC/MAX 统计上就不行"这个错前提在仓库里活了一个多月——而它正是"某因子该不该
# 重审"的直接依据。读数只准有一份源(methodology §5/§6/§10),代码只准复述:
# 文档改了这里先红(逼同步),旧字面量复活这里也红(防回退)。

REPO = Path(__file__).resolve().parents[1]
AUTHORITATIVE_NW_T = {"EP": 6.59, "BP": 9.16, "IVOL": -17.02, "ACC": 4.05,
                      "MOM": -2.20, "MAX": -14.08, "NLIMIT": -11.66, "TURN": -9.14}
AUTHORITATIVE_INCREMENT_T = {"ACC": 1.63, "MAX": 0.73, "NLIMIT": 0.74}
# 2026-07-10 修复前的旧读数字面量,不得再出现在生产源码里(本文件是白名单——
# 旧值就存在这里当靶子)
SUPERSEDED_LITERALS = ("t4.36", "t7.14", "t-14.7", "t2.36", "t 2.36",
                       "+0.34%", "-0.28%", "多头腿≈0")
CITING_SOURCES = ("scripts/factor_rank.py", "ashare_gauntlet/factor_model.py",
                  "scripts/factor_tearsheet.py")


def _table_nw_t(section: str, wanted: dict[str, float]) -> dict[str, float]:
    """取 methodology 管道表的 NW t 列(第 3 列),只收 wanted 里的因子名。"""
    out: dict[str, float] = {}
    for line in section.splitlines():
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] in wanted:
            try:
                out[cells[0]] = round(float(cells[2].replace("−", "-").replace("+", "")), 2)
            except ValueError:
                continue
    return out


def test_methodology_still_carries_the_readings_the_code_quotes():
    doc = (REPO / "docs" / "methodology.md").read_text(encoding="utf-8")
    five_six = doc.split("## 5.")[1].split("## 7.")[0]
    assert _table_nw_t(five_six, AUTHORITATIVE_NW_T) == AUTHORITATIVE_NW_T
    increments = doc.split("毛增量/期")[1].split("四者全军覆没")[0]
    assert _table_nw_t(increments, AUTHORITATIVE_INCREMENT_T) == AUTHORITATIVE_INCREMENT_T


def test_no_production_source_quotes_pre_20260710_readings():
    for rel in CITING_SOURCES:
        text = (REPO / rel).read_text(encoding="utf-8")
        stale = [lit for lit in SUPERSEDED_LITERALS if lit in text]
        assert stale == [], f"{rel} 仍在引用 2026-07-10 数据修复前的读数:{stale}"


def test_factor_rank_states_the_real_reason_acc_and_max_are_out():
    # "不入分"的理由必须落在增量门上而不是"统计不行":ACC 五门全过(t+4.05)、被
    # X-02 增量 t+1.63 拦下,这两个数同时出现才算把理由写对
    src = (REPO / "scripts" / "factor_rank.py").read_text(encoding="utf-8").replace("−", "-")
    for lit in ("6.59", "9.16", "17.02", "+0.29%", "4.05", "1.63", "-2.20"):
        assert lit in src, f"factor_rank.py 未复述权威读数 {lit}"