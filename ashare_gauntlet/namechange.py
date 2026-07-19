"""namechange PIT 名称面板(X-05)——历史逐日证券名称/ST 状态的时点还原。

composite 回测的 ST 剔除此前用 stock_basic **最终名称**近似(评审三轮起挂账的
残余偏差:后来戴帽的被错误早剔、当时戴帽后摘帽的被错误纳入)。本模块用
tushare `namechange` 的名称变更区间还原任意 asof 日的生效名称:

- **PIT 依据 = start_date(变更生效日)**:名称在生效日起就挂在行情终端上,
  市场参与者当日可见——用生效日区间不引入任何前视(ann_date 只是公告时点,
  生效前的名称仍是旧名)。
- namechange 无记录的票 = 上市以来名称从未变过,现名即历史名(fallback 语义)。
- 有记录但 asof 早于其最早 start_date 的票在 name_asof 中缺席,st_codes_asof
  会将其 **fallback 到现名**(即该窗口退化回最终名称近似)。覆盖实测
  (2026-07-19):初始名在表内的 5,802/5,864 只,存在无覆盖窗口的 62 只全部为
  北交所(主板宇宙外;唯一现名含 ST 者 920090.BJ),完全无记录 2 只均非主板
  ——对沪深主板回测该退化路径实际命中 0;若宇宙扩到北交所须先补该缺口。

缓存:data/cache/namechange/all.parquet(scripts.backfill_namechange 拉取)。
"""
from __future__ import annotations

import os

import pandas as pd


def load_namechange(cache_dir: str) -> pd.DataFrame:
    """读 namechange 缓存并校验。缺文件/无效 start_date 一律 fail-loud。

    start_date 无效的行无法定位到时间轴上,静默丢弃=该票该段历史悄悄退化回
    最终名称近似——拒绝;脏行应在 backfill 层面对着数字清理并记录。
    """
    path = os.path.join(cache_dir, "namechange", "all.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 不存在——先跑 python -m scripts.backfill_namechange(X-05)")
    df = pd.read_parquet(path)
    need = {"ts_code", "name", "start_date"}
    if not need.issubset(df.columns):
        raise ValueError(f"namechange 缓存缺列:{sorted(need - set(df.columns))}")
    bad = ~df["start_date"].astype(str).str.fullmatch(r"\d{8}", na=True) | df["start_date"].isna()
    if bool(bad.any()):
        sample = df.loc[bad, ["ts_code", "name", "start_date"]].head(5).to_dict("records")
        raise ValueError(f"namechange 含 {int(bad.sum())} 行无效 start_date(如 {sample})——"
                         f"回 backfill 层清理,不静默丢")
    return df


def name_asof(changes: pd.DataFrame, asof: str) -> pd.Series:
    """每股 asof 当日生效名称:start_date<=asof 的最近一次变更(PIT)。

    **名称链连续假设(end_date 不参与判定)**:证券任何时刻都挂着一个名称,
    上一名称自然延续到下一次变更生效——故只用生效日链;tushare 的 end_date
    与下条 start_date 间存在重叠/空档噪声,且末条(现行名)end_date 为空,
    用它判"区间已结束"反而制造无名称真空(Codex review 澄清)。
    输入乱序亦可(内部稳定排序);同 (ts_code, start_date) 多行取末行(确定性)。
    asof 早于某票全部记录 → 该票缺席(调用方按 fallback 语义处理)。
    """
    vis = changes[changes["start_date"].astype(str) <= str(asof)]
    if vis.empty:
        return pd.Series(dtype=object)
    vis = vis.sort_values(["ts_code", "start_date"], kind="mergesort")
    return vis.groupby("ts_code")["name"].last()


def st_codes_asof(changes: pd.DataFrame, asof: str, fallback: pd.Series) -> set[str]:
    """asof 当日名称含 "ST" 的票集合(与生产剔除同一定义性规则,含 *ST/S*ST)。

    fallback = stock_basic 现名(index=ts_code,定义 universe):namechange 无记录
    的票名称从未变过,现名即历史名;有记录且 asof 已被区间覆盖的票以 PIT 名称
    覆盖现名;有记录但 asof 早于最早 start_date 的票同样落到现名 fallback
    (已知退化路径,主板命中 0——见模块 docstring 覆盖实测)。
    """
    nm = name_asof(changes, asof)
    combined = nm.reindex(fallback.index).fillna(fallback.astype(str))
    hit = combined.astype(str).str.contains("ST", na=False)
    return set(combined.index[hit].astype(str))
