"""治理雷结构化:pledge_detail(股东质押明细)与 fina_audit(审计意见)纯函数层。

预警表 ⛔ 只抓公司层粗雷(pledge_stat 整体质押/已完成减持/亏损),抓不到**控股股东
个人质押**与**非标审计意见**这两类治理红旗 —— 本模块把这两张表结构化。只报数不设
阈值:"质押占持股比多高算危险"是人工/factcheck 的判断,不往脚本塞 magic number。

口径以实测为准(2026-07-02 对 002602.SZ / 600989.SH 探字段):
- pledge_detail 的解押**不回写**旧行 is_release,而是另生成 is_release==1 的新行,
  所以对 is_release==0 求和会重复计数(实测王佶 sum≈267530 万股 vs 真实 75463),
  部分解押还会漏计(实测永丰国际 sum=0 但快照 2500 万股)。未解押的唯一正确口径 =
  每股东最新 ann_date 行的 pledged_amount(该股东质押总量快照,实测全解押后归 0)。
- h_total_ratio 实测 = 持股总数占总股本比(%),**不是**"质押占其持股比"
  (实测王佶 h_total_ratio=10.36 而 75462.78/76404.56=98.8%);
  质押占其持股比 = pledged_amount / holding_amount(同一公告快照的定义性比值)。
"""

from typing import Any

import pandas as pd

# 监管术语:年报审计意见的唯一"干净"类型。任何其它措辞(保留意见/无法表示意见/
# 否定意见/带强调事项段的无保留意见)都属非标 —— 定义性判断,非阈值。
STANDARD_OPINION = "标准无保留"


def _num(value: Any) -> float | None:
    """数值字段安全转 float;NaN/缺失 → None(不伪造 0)。与 fundamentals._num 同约定。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def controller_pledge(pledge_detail: pd.DataFrame | None) -> dict[str, Any]:
    """未解押质押占其持股比最高的股东(pledge_detail 全表 → 一个股东快照)。

    每个 holder_name 取最新 ann_date 行读快照字段(实测同日多行快照一致):
    - pledged_amount   = 该股东未解押质押总量(万股,全解押后实测归 0 → 跳过)
    - 占其持股比        = pledged_amount / holding_amount × 100(任一缺 → None 不伪造)
    - 占总股本比        = 占其持股比 × h_total_ratio / 100(h_total_ratio 缺 → None)

    返回 ``{holder_name, pledged_ratio_of_holding, ratio_of_total, asof,
    pledged_amount_wan, holding_amount_wan}``;空表/全解押 → ``{}``。
    比例可算者优先参与"最高"排序;比例未知但有未解押量的股东不静默丢弃
    (未知 ≠ 无风险),在无可算者时兜底返回(比例字段为 None)。
    """
    if pledge_detail is None or pledge_detail.empty:
        return {}
    df = pledge_detail.copy()
    df["ann_date"] = df["ann_date"].astype(str)

    candidates: list[dict[str, Any]] = []
    for holder, group in df.groupby("holder_name"):
        asof = group["ann_date"].max()
        row = group[group["ann_date"] == asof].iloc[0]
        pledged = _num(row.get("pledged_amount"))
        if pledged is not None and pledged == 0:
            continue  # 最新快照已全解押 → 当前无未解押质押
        holding = _num(row.get("holding_amount"))
        h_ratio = _num(row.get("h_total_ratio"))  # 持股总数占总股本比(%)
        ratio_of_holding = (
            pledged / holding * 100.0 if pledged is not None and holding else None
        )
        ratio_of_total = (
            ratio_of_holding * h_ratio / 100.0
            if ratio_of_holding is not None and h_ratio is not None
            else None
        )
        candidates.append({
            "holder_name": str(holder),
            "pledged_ratio_of_holding": ratio_of_holding,
            "ratio_of_total": ratio_of_total,
            "asof": str(asof),
            "pledged_amount_wan": pledged,
            "holding_amount_wan": holding,
        })

    if not candidates:
        return {}

    def _key(c: dict[str, Any]) -> tuple[bool, float, float]:
        ratio = c["pledged_ratio_of_holding"]
        pledged = c["pledged_amount_wan"]
        return (
            ratio is not None,                                   # 可算比例者优先
            ratio if ratio is not None else float("-inf"),       # 比例最高
            pledged if pledged is not None else float("-inf"),   # 同比例取质押量大者
        )

    return max(candidates, key=_key)


def audit_opinion(fina_audit: pd.DataFrame | None) -> dict[str, Any]:
    """最新报告期审计意见(fina_audit 全表 → 一期结论)。

    最新期 = end_date 最大;同期多次公告(更正)以 ann_date 最新为准。
    is_nonstandard 定义性:audit_result 不含"标准无保留"即 True ——
    保留意见/无法表示意见/否定意见/带强调事项段的无保留意见都是治理红旗;
    audit_result 缺失也按 True 上报(缺失 ≠ 干净,surface 给人工查)。
    空表 → ``{}``。
    """
    if fina_audit is None or fina_audit.empty:
        return {}
    df = fina_audit.copy()
    df["end_date"] = df["end_date"].astype(str)
    if "ann_date" in df.columns:
        df["ann_date"] = df["ann_date"].astype(str)
        df = df.sort_values(["end_date", "ann_date"])
    else:
        df = df.sort_values("end_date")
    row = df.iloc[-1]
    raw = row.get("audit_result")
    result = None if raw is None or pd.isna(raw) else str(raw)
    return {
        "end_date": str(row["end_date"]),
        "audit_result": result,
        "is_nonstandard": not (result is not None and STANDARD_OPINION in result),
    }
