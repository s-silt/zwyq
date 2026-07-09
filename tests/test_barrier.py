"""止盈止损路径引擎(用户质疑"按月持有不现实,到点位就止盈"的验证实验)。

逐日走查:入场价=T+1开盘;每日先看跳空(开盘已越界→按开盘成交,止损跳空更亏、
止盈跳空更赚,真实);再看盘中触碰(按触发价成交);同日双触按**止损优先**
(日内先后无从得知,取对策略不利的保守假设);停牌日(NaN)跳过;到期未触
→ 期末收盘离场;数据尽头(退市)→ 最后有效收盘。
"""
import numpy as np
import pytest

from scripts.barrier_experiment import barrier_paths


def _panel(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]   # o, h, l, c 各为 (T,1)


def test_take_profit_hit_intraday():
    o, h, l, c = _panel([[10.0, 10.7, 9.9, 10.5],
                         [10.5, 11.1, 10.4, 10.9]])     # 次日盘中触 +8%(10.8)
    px, days, why = barrier_paths(np.array([10.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert px[0] == pytest.approx(10.8) and why[0] == "tp" and days[0] == 1


def test_stop_loss_hit_intraday():
    o, h, l, c = _panel([[10.0, 10.2, 9.4, 9.6]])       # 当日盘中触 -5%(9.5)
    px, days, why = barrier_paths(np.array([10.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert px[0] == pytest.approx(9.5) and why[0] == "sl" and days[0] == 0


def test_both_touched_same_day_stop_wins():
    o, h, l, c = _panel([[10.0, 11.0, 9.0, 10.0]])      # 同日双触 → 保守按止损
    px, days, why = barrier_paths(np.array([10.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert why[0] == "sl" and px[0] == pytest.approx(9.5)


def test_gap_down_exits_at_open_worse_than_stop():
    o, h, l, c = _panel([[10.0, 10.1, 9.9, 10.0],
                         [9.0, 9.2, 8.8, 9.1]])          # 次日跳空低开 9.0 < 9.5
    px, days, why = barrier_paths(np.array([10.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert px[0] == pytest.approx(9.0) and why[0] == "sl"   # 按开盘 9.0 成交,更亏(真实)


def test_gap_up_exits_at_open_better_than_tp():
    o, h, l, c = _panel([[10.0, 10.1, 9.9, 10.0],
                         [11.0, 11.2, 10.9, 11.1]])
    px, days, why = barrier_paths(np.array([10.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert px[0] == pytest.approx(11.0) and why[0] == "tp"


def test_time_stop_exit_at_last_close():
    o, h, l, c = _panel([[10.0, 10.2, 9.8, 10.1],
                         [10.1, 10.3, 9.9, 10.0],
                         [10.0, 10.2, 9.8, 10.05]])      # 3日都没触 → 期末收盘
    px, days, why = barrier_paths(np.array([10.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert px[0] == pytest.approx(10.05) and why[0] == "time" and days[0] == 2


def test_suspension_days_skipped_then_trigger():
    o, h, l, c = _panel([[10.0, 10.1, 9.9, 10.0],
                         [np.nan, np.nan, np.nan, np.nan],   # 停牌
                         [9.0, 9.1, 8.9, 9.0]])              # 复牌跳空 → 开盘 9.0 走
    px, days, why = barrier_paths(np.array([10.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert px[0] == pytest.approx(9.0) and why[0] == "sl"


def test_delisted_mid_window_exit_last_valid_close():
    o, h, l, c = _panel([[10.0, 10.2, 9.8, 9.9],
                         [np.nan, np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, np.nan]])  # 从此无价(退市)
    px, days, why = barrier_paths(np.array([10.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert px[0] == pytest.approx(9.9) and why[0] == "end"


def test_vectorized_multiple_stocks():
    o = np.array([[10.0, 20.0], [10.5, 19.2]])
    h = np.array([[10.1, 20.2], [11.0, 19.4]])
    l = np.array([[9.9, 19.8], [10.4, 18.5]])
    c = np.array([[10.0, 20.0], [10.9, 19.0]])
    px, days, why = barrier_paths(np.array([10.0, 20.0]), o, h, l, c, tp=0.08, sl=0.05)
    assert why[0] == "tp" and px[0] == pytest.approx(10.8)   # 股1 次日盘中触止盈
    assert why[1] == "sl" and px[1] == pytest.approx(19.0)   # 股2 次日盘中触止损(开盘未跳空)
