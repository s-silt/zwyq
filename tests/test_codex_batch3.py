"""Codex review 第三批(2026-07-09,范围 88ccad3..HEAD)修复测试。

P1×5:①pct_change 默认 ffill 把停牌日变 0% 收益(停牌股伪装低波进 composite);
②ivol_capm 不查单股有效样本数(新股 4 个点也出 IVOL,绕过 21 日门槛);
③NLIMIT 回测缺 up_limit 静默 False vs 生产 fail-loud(门禁与生产不是同一对象);
④tearsheet 多头腿成本用两腿平均 τ + 准入缺"多头腿净>0"门(MOM 教训写了没执行);
⑤backfill_delisted VIP 空页落缓存(空 parquet 永久挡住重拉)。
P2:腿集合含 entry NaN 不可执行股;哨兵单字段异常崩整批;注释残留。
"""
import math

import numpy as np
import pandas as pd
import pytest

from ashare_gauntlet.factor_model import daily_returns, ivol_capm
from ashare_gauntlet.intraday import parse_tencent_quote


# ---------- ① 停牌 NaN 不得变 0% 收益 ----------

def test_daily_returns_suspension_stays_nan():
    # 价格面板:b 股中间停牌 2 日(NaN)——默认 pct_change 会 ffill 成 0% 收益,
    # 把停牌股伪装成低波动票;必须保持 NaN(复牌日收益按跨停牌区间计算也为 NaN,
    # fill_method=None 语义)
    px = pd.DataFrame({"a": [10.0, 10.1, 10.2, 10.3],
                       "b": [20.0, float("nan"), float("nan"), 21.0]})
    r = daily_returns(px)
    assert math.isnan(r["b"].iloc[1]) and math.isnan(r["b"].iloc[2])
    assert math.isnan(r["b"].iloc[3])          # 复牌日:前值 NaN → 收益无定义,不伪造
    assert r["a"].iloc[1] == pytest.approx(0.01)


# ---------- ② IVOL 单股有效样本门槛 ----------

def test_ivol_capm_insufficient_valid_pairs_nan():
    # 面板总长够 21,但 b 股只有 4 个有效收益(新股/长停牌)→ 必须 NaN,
    # 否则 4 个点的"低波"绕过入池门槛
    rng = np.random.default_rng(3)
    n = 30
    a = rng.normal(0, 0.01, n)
    b = np.full(n, np.nan)
    b[-4:] = rng.normal(0, 0.001, 4)            # 仅 4 个有效点且波动极小
    ret = pd.DataFrame({"a": a, "b": b})
    mkt = pd.Series(rng.normal(0, 0.01, n))
    out = ivol_capm(ret, mkt, window=21)
    assert math.isfinite(out["a"])
    assert math.isnan(out["b"])


# ---------- ③ NLIMIT 触板行:缺 up_limit 必须显式 NaN(不许静默 False) ----------

def test_touched_row_missing_limit_is_nan():
    from scripts.factor_backtest import touched_row
    hi = pd.Series({"a": 11.0, "b": 9.0, "c": float("nan")})   # c=当日停牌无价
    ul = pd.Series({"a": 11.0})                                 # b 缺涨停价
    row = touched_row(hi, ul)
    assert row["a"] == 1.0                       # 触板
    assert math.isnan(row["b"])                  # 缺规则价 → NaN(生产同口径 fail 语义)
    assert math.isnan(row["c"])                  # 停牌无价 → NaN


# ---------- ④ 准入第五门:多头腿成本后>0(逐腿 τ) ----------

def test_admission_verdict_requires_positive_long_leg():
    from scripts.factor_tearsheet import admission_verdict
    ok, reasons = admission_verdict(full_t=3.5, real_net=0.002,
                                    loyo={"2020": 3.1, "2021": 3.4},
                                    up_ic=0.03, down_ic=0.02, leg_net=0.001)
    assert ok
    bad, reasons = admission_verdict(full_t=3.5, real_net=0.002,
                                     loyo={"2020": 3.1, "2021": 3.4},
                                     up_ic=0.03, down_ic=0.02, leg_net=-0.001)
    assert not bad and any("多头腿" in r for r in reasons)


def test_long_leg_net_uses_matching_leg_turnover():
    # 反转向因子(IC<0)→ 多头腿=低分位腿,成本必须用 TOLO 而非两腿平均
    from scripts.factor_tearsheet import long_leg_net
    res = pd.DataFrame({
        "IC_Y": [-0.05] * 3,
        "QLO_Y": [0.02] * 3, "QHI_Y": [-0.01] * 3, "mkt_fwd": [0.005] * 3,
        "TOLO_Y": [0.1] * 3, "TOHI_Y": [0.9] * 3, "cost_rt": [0.004] * 3,
    })
    net = long_leg_net(res, "Y")
    # 低腿超额 0.015 − 0.1×0.004 = 0.0146(若误用平均 τ=0.5 则 0.013)
    assert net == pytest.approx(0.015 - 0.1 * 0.004)


# ---------- ⑤ VIP 空页不得落缓存 ----------

def test_require_nonempty_period_raises():
    from scripts.backfill_delisted import require_nonempty
    with pytest.raises(RuntimeError, match="income_vip"):
        require_nonempty(pd.DataFrame(), "income_vip", "20160331")
    df = pd.DataFrame({"ts_code": ["x"]})
    assert require_nonempty(df, "income_vip", "20160331") is df


# ---------- P2:哨兵单字段异常跳行不崩批 ----------

def test_parse_tencent_quote_junk_numeric_field_skips_line():
    good = ('v_sh600875="1~东方电气~600875~27.71~28.40~28.30' + "~x" * 26 + '~-2.43~~~";')
    junk = ('v_sz000589="1~贵州轮胎~000589~--~4.30~4.28' + "~x" * 26 + '~--~~~";')
    q = parse_tencent_quote(good + "\n" + junk)
    assert "600875.SH" in q and "000589.SZ" not in q


def test_trend_shared_between_production_and_backtest():
    # TREND 升展示列(用户批准):生产/回测共用同一实现(ivol_capm 先例),
    # 从 factor_model 导入;factor_backtest 保持 re-export 兼容
    from ashare_gauntlet.factor_model import trend_ma_distance as fm_trend
    from scripts.factor_backtest import trend_ma_distance as bt_trend
    assert fm_trend is bt_trend
