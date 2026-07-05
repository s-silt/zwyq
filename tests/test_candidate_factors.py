"""P1 交易行为族候选因子(吸纳终榜 4/5/8:TURN/ILLIQ/MAX/涨停次数/IVOL)。

全部先过 IC 门禁再谈入分(准入纪律:NW t>3+成本后>0+方向稳定+年度切片)。
窗口约定:月=21、年=252 个交易日(与 MOM_LB/MOM_SKIP 同一定义性惯例,非新常数)。
出处:MAX=Bali-Cakici-Whitelaw 2011 彩票异象(中国显著,Anomalies in China);
IVOL=Ang et al. 2006 特异波动(市场模型残差口径,显式标注非 FF3 残差);
ILLIQ=Amihud 2002(月窗为 GKX/LWZ 特征工程惯例,原式年窗);
TURN=异常换手 ln(月均/年均)(LWZ 2022:中国横截面最强特征族);
NLIMIT=近月触涨停次数(交易所规则价锚,扩展现有 ⚡5日标注)。
"""
import math

import numpy as np
import pandas as pd

from scripts.factor_backtest import amihud_illiq, ivol_capm, max_daily_ret, turn_abnormal

RNG = np.random.default_rng(7)


def _ret_panel(n_days: int = 30, cols: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    return pd.DataFrame(RNG.normal(0, 0.01, (n_days, len(cols))), columns=list(cols))


# ---------- MAX:近月最大单日收益(彩票) ----------

def test_max_daily_ret_picks_window_max():
    ret = _ret_panel(30)
    ret.iloc[25, 0] = 0.095                       # a 在窗内有一天+9.5%
    out = max_daily_ret(ret, window=21)
    assert abs(out["a"] - 0.095) < 1e-12
    assert out["b"] < 0.05


def test_max_daily_ret_insufficient_history_nan():
    out = max_daily_ret(_ret_panel(10), window=21)  # 不足21日 → NaN 不伪造
    assert math.isnan(out["a"]) and math.isnan(out["b"])


# ---------- IVOL:市场模型残差波动 ----------

def test_ivol_capm_zero_for_pure_beta_stock():
    mkt = pd.Series(RNG.normal(0, 0.01, 30))
    ret = pd.DataFrame({"pure": 2.0 * mkt, "noisy": mkt + RNG.normal(0, 0.02, 30)})
    out = ivol_capm(ret, mkt, window=21)
    assert out["pure"] < 1e-10                    # 纯β股:残差为零
    assert out["noisy"] > 0.01                    # 有特异波动

def test_ivol_capm_insufficient_nan():
    mkt = pd.Series(RNG.normal(0, 0.01, 10))
    out = ivol_capm(_ret_panel(10), mkt, window=21)
    assert math.isnan(out["a"])


# ---------- ILLIQ:Amihud 月窗 ----------

def test_amihud_illiq_higher_for_thin_stock():
    ret = pd.DataFrame({"thick": [0.01] * 21, "thin": [0.01] * 21})
    amt = pd.DataFrame({"thick": [1e9] * 21, "thin": [1e6] * 21})   # 千倍成交额差
    out = amihud_illiq(ret, amt, window=21)
    assert out["thin"] > out["thick"] * 100


def test_amihud_illiq_zero_amount_day_excluded():
    # 停牌日 amount=0/NaN:不作分母(inf 污染),按 skipna 均值
    ret = pd.DataFrame({"a": [0.01] * 21})
    amt = pd.DataFrame({"a": [1e8] * 20 + [0.0]})
    out = amihud_illiq(ret, amt, window=21)
    assert math.isfinite(out["a"])


# ---------- TURN:异常换手 ln(月均/年均) ----------

def test_turn_abnormal_log_ratio():
    tr = pd.DataFrame({"hot": [1.0] * 252 + [3.0] * 21,     # 近月换手 3 倍于常态
                       "cold": [1.0] * 252 + [0.5] * 21})
    out = turn_abnormal(tr, month=21, year=252)
    assert abs(out["hot"] - math.log(3.0)) < 1e-9
    assert abs(out["cold"] - math.log(0.5)) < 1e-9


def test_turn_abnormal_insufficient_history_nan():
    tr = pd.DataFrame({"a": [1.0] * 100})                   # 不足 252+21
    assert math.isnan(turn_abnormal(tr, month=21, year=252)["a"])
