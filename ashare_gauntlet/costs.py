"""A股交易成本模型 —— PIT(point-in-time)分段费率,回测/前向收益"成本后口径"的唯一入口。

五路文献研读(Qlib/QuantsPlaybook/RQAlpha/LWZ)一致判定:无成本回测的前向收益系统性
虚高,成本模型是所有边际因子取舍的前提。本模块只收录**监管常数与行业惯例**:

- 印花税分段(STAMP_DUTY_SEGMENTS):财政部/税务总局公告的法定税率,按交易日 PIT 取段,
  不能用单一固定值(2023-08-28 减半,横跨回测样本期);
- 5元最低佣金(MIN_COMMISSION_CNY):券商经纪合同普遍的"最低5元"条款,行业惯例常数。

**commission_rate / slippage_rate 不在本库写默认值**:佣金是用户与券商的合同参数,
滑点是市场微观结构的实测参数,都不是库常数——由调用方传入并在调用处注明出处
(如 scripts.factor_backtest 的 --commission/--slippage 默认值及其文献来源)。
"""
from __future__ import annotations

import math
import re

# 证券交易印花税(卖出单边)PIT 分段表:(生效起始日 YYYYMMDD, 税率),按生效日升序。
# 写成分段表而非 if/else,未来税率再调整只需追加一行(监管常数,来源见各行):
STAMP_DUTY_SEGMENTS: tuple[tuple[str, float], ...] = (
    # 2008-09-19 起,经国务院批准,财政部/国家税务总局将证券交易印花税由双边征收
    # 改为**出让方单边**征收,税率 1‰(2008-09-18 发布)。更早的双边/其它税率 regime
    # 未录入——本库所有含价格缓存均晚于此,遇更早日期 fail-loud 拒绝外推。
    ("20080919", 0.001),
    # 2023-08-28 起证券交易印花税**减半征收** → 0.5‰
    # (财政部 税务总局公告 2023年第39号《关于减半征收证券交易印花税的公告》)。
    ("20230828", 0.0005),
)

# 券商行业"最低5元"收费惯例(经纪合同普遍条款,非监管定价但事实上全行业一致)。
# 注意:对 ≤1万元 的小仓位,5元地板占比 ≥0.05%(5/10000)单边,一次完整买卖 ≥0.1%,
# 远高于名义万2.5佣金——双仓制短线仓(≤1万,见交易约束)受此影响最直接。
MIN_COMMISSION_CNY: float = 5.0

_SIDES = ("buy", "sell")
_DATE_RE = re.compile(r"^\d{8}$")


def _require_yyyymmdd(trade_date: str) -> None:
    """全库日期约定为 'YYYYMMDD' 纯数字串(缓存分区/交易日历同款);其它格式 fail-loud。"""
    if not isinstance(trade_date, str) or not _DATE_RE.match(trade_date):
        raise ValueError(
            f"trade_date 需为 'YYYYMMDD' 纯数字串(全库缓存分区/交易日历同约定),得到 {trade_date!r}")


def stamp_duty_rate(trade_date: str) -> float:
    """交易日 ``trade_date`` 当期的印花税率(卖出单边,PIT 分段)。

    日期早于分段表最早段(2008-09-19 单边征收起点)→ fail-loud:更早处于双边/其它
    税率 regime,未录入,静默外推会低估历史成本。
    """
    _require_yyyymmdd(trade_date)
    earliest = STAMP_DUTY_SEGMENTS[0][0]
    if trade_date < earliest:
        raise ValueError(
            f"trade_date={trade_date} 早于印花税分段表最早段 {earliest}(单边征收起点)"
            f"——更早历史处于双边/其它税率 regime 未录入,拒绝外推")
    rate = STAMP_DUTY_SEGMENTS[0][1]
    for start, r in STAMP_DUTY_SEGMENTS:
        if trade_date >= start:
            rate = r
    return rate


def stamp_duty(trade_date: str, sell_amount: float) -> float:
    """卖出金额 ``sell_amount``(元)在 ``trade_date`` 应缴印花税额(元,卖出单边)。"""
    if not isinstance(sell_amount, (int, float)) or sell_amount < 0 or math.isnan(sell_amount):
        raise ValueError(f"sell_amount 需为非负金额(元),得到 {sell_amount!r}")
    return sell_amount * stamp_duty_rate(trade_date)


def _require_nonneg_rates(commission_rate: float, slippage_rate: float) -> None:
    # isfinite 同时拦 NaN/±inf:NaN 费率会让成本后列整列静默变 NaN(外部 review 点名)
    for name, r in (("commission_rate", commission_rate), ("slippage_rate", slippage_rate)):
        if not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0:
            raise ValueError(
                f"{name} 需为非负有限数,得到 {r!r}——负/NaN/inf 费率无经济含义,多半是调用方传参错误")


def round_trip_cost_rate(trade_date: str, commission_rate: float, slippage_rate: float,
                         sell_date: "str | None" = None) -> float:
    """单次**完整买卖**(一买一卖)的总费率 = 2×佣金 + 2×滑点 + 印花税(卖出侧)。

    commission_rate(单边佣金率)与 slippage_rate(单边滑点率)由调用方传入并注明出处
    (用户合同/实测参数,非库常数)。印花税按**卖出日** ``sell_date`` PIT 取段——
    税是卖出时缴的,持有期跨税改日(如 2023-08-28 减半)时用买入/信号日取段会错扣
    (外部 review 点名);``sell_date`` 缺省回退 ``trade_date``(当日往返/未知卖出日
    的保守近似)。sell_date 早于 trade_date 无经济含义,fail-loud。
    比率口径忽略 5 元佣金地板(地板依赖绝对金额,见 :func:`trade_cost`)——对小仓位
    该口径**低估**成本,大额组合层面无碍。
    """
    _require_nonneg_rates(commission_rate, slippage_rate)
    sd = trade_date if sell_date is None else sell_date
    _require_yyyymmdd(sd)
    if sd < trade_date:
        raise ValueError(f"sell_date={sd} 早于 trade_date={trade_date}——卖先于买无经济含义")
    return 2.0 * commission_rate + 2.0 * slippage_rate + stamp_duty_rate(sd)


def trade_cost(trade_date: str, side: str, amount: float, commission_rate: float,
               slippage_rate: float, min_commission: float = MIN_COMMISSION_CNY) -> dict[str, float]:
    """单边一笔交易的成本分解(元):{commission, stamp, slippage, total}。

    - commission = max(amount×commission_rate, min_commission):5元地板为券商行业最低
      收费惯例;对 ≤1万元 仓位地板占比可达 0.05%+ 单边(5/10000),小仓位实际费率远高于
      名义佣金率——双仓制短线仓(≤1万)据此评估;
    - stamp:仅 side='sell' 计(印花税卖出单边,PIT 分段);
    - slippage = amount×slippage_rate(线性冲击近似,参数出处由调用方注明);
    - side 非 'buy'|'sell'、amount 非正、费率为负 → fail-loud。
    """
    if side not in _SIDES:
        raise ValueError(f"side 需为 'buy'|'sell',得到 {side!r}")
    if not isinstance(amount, (int, float)) or not amount > 0:   # not> 同时拦 NaN
        raise ValueError(f"amount 需为正的成交金额(元),得到 {amount!r}")
    _require_nonneg_rates(commission_rate, slippage_rate)
    commission = max(amount * commission_rate, min_commission)
    stamp = stamp_duty(trade_date, amount) if side == "sell" else 0.0
    slippage = amount * slippage_rate
    return {"commission": commission, "stamp": stamp, "slippage": slippage,
            "total": commission + stamp + slippage}
