"""门禁证据体检(纯函数,只读):准入证据有没有在静默变旧/退化。

补的缺口(跨层审计 Q4 + 深读 R1 头号建议):五门/组合 t 的结论冻结在一次复跑上,
`factor_backtest`/`factor_tearsheet` 不在 eod_ops 里、无排期再验证;"持续检验交给
pick_track 月度"这条承诺在口径上不成立(pick_track 是"快照日→今日"的重叠累计、对
沪深300、只扣一次成本),已被删除——于是**当前没有任何可执行的复审触发器**。

本模块是主刹车(政策评审 Q4 的 D 方案):**盯"准入证据是否退化"而非"短窗收益是否为负"**。
后者在低波价值的风格逆风期会常亮(composite 逐年净超额本就有多年为负或≈0),常亮的
警报等于没有警报;前者直接问"当初让它入分的证据现在还成立吗"。

零新判据:阈值全部复用 factor_tearsheet 的 T_ADMIT(3.0)/T_FOLD(2.0)与
admission_verdict 五门本身,本模块不定义任何新常数、不改任何研究口径,只做
"当前读数 vs 冻结基线"的比对与呈现。判定只触发**人工复审**,绝不自动改因子。

import 阶段无 I/O 副作用。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

# 生产入分因子(与 scripts/factor_rank.COMPOSITE_FACTORS 同源,去掉 f_ 前缀后即回测列名)
PRODUCTION_FACTORS = ("EP", "BP", "IVOL")


class GateHealthError(ValueError):
    """输入读数结构非法——不猜不静默。"""


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def factor_gate_status(detail_rows: Sequence[dict[str, Any]],
                       factor: str) -> dict[str, Any]:
    """对单个入分因子跑一遍**既有五门**,返回判定与各门读数。

    复用 factor_tearsheet 的实现(同一份判据,不复制逻辑):full NW t / 可交易向真实净 /
    LOYO 各折 / 涨跌市 IC / 多头腿成本后。缺列 → status=NO_DATA(不当作通过)。
    """
    import pandas as pd

    from scripts.factor_tearsheet import (
        admission_verdict,
        long_leg_net,
        loyo_tstats,
        state_split,
        tradable_real_net,
    )
    from ashare_gauntlet.backtest import newey_west_tstat

    res = pd.DataFrame(list(detail_rows))
    ic_col = f"IC_{factor}"
    if ic_col not in res.columns or "mkt_fwd" not in res.columns:
        return {"factor": factor, "status": "NO_DATA", "passed": None,
                "reasons": [f"读数缺 {ic_col} 或 mkt_fwd 列——无法体检,不视为通过"]}
    ic = res[ic_col].dropna()
    if ic.empty:
        return {"factor": factor, "status": "NO_DATA", "passed": None,
                "reasons": [f"{ic_col} 全为空"]}
    _, full_t, _ = newey_west_tstat(ic)
    real_net = tradable_real_net(res, factor) if f"TO_{factor}" in res.columns else float("nan")
    loyo = loyo_tstats(res, ic_col)
    up_ic, down_ic = state_split(res, ic_col)
    leg_net = long_leg_net(res, factor) if f"QLO_{factor}" in res.columns else float("nan")
    passed, reasons = admission_verdict(full_t, real_net, loyo, up_ic, down_ic, leg_net)
    return {
        "factor": factor, "status": "PASS" if passed else "FAIL", "passed": passed,
        "reasons": reasons, "n": int(len(ic)),
        "nw_t": round(float(full_t), 3), "real_net": round(float(real_net), 6),
        "leg_net": round(float(leg_net), 6),
        "loyo_min_abs_t": round(min((abs(t) for t in loyo.values()), default=float("nan")), 3),
        "up_ic": round(float(up_ic), 4), "down_ic": round(float(down_ic), 4),
    }


def composite_nw_t(rows: Sequence[dict[str, Any]], port: str = "PROD") -> dict[str, Any]:
    """组合级 NW t。

    **口径对齐(务必保持)**:`nw_t` = **毛超额**(ret − mkt_fwd)的 NW t,与
    composite_backtest 报告表里的 "NW t" 列、以及 methodology §10 引用的那个数**同口径**
    (该处实现见 composite_backtest.py 的 `newey_west_tstat(ex.dropna())`)。
    另附 `nw_t_net` = 净超额(再减 τ×cost)的 t 作参考——两者是不同的数,**不可混用**:
    贴错标签会让基线比对拿两个口径的值做差,凭空造出"退化"或掩盖真退化。
    """
    import pandas as pd

    from ashare_gauntlet.backtest import newey_west_tstat

    res = pd.DataFrame(list(rows))
    need = {f"ret_{port}", "mkt_fwd", f"TO_{port}", "cost_rt"}
    if not need <= set(res.columns):
        return {"port": port, "status": "NO_DATA", "nw_t": None, "n": 0,
                "note": f"读数缺 {sorted(need - set(res.columns))}——不体检,不视为健康"}
    gross = (res[f"ret_{port}"] - res["mkt_fwd"]).dropna()
    if gross.empty:
        return {"port": port, "status": "NO_DATA", "nw_t": None, "n": 0, "note": "毛超额全空"}
    _, t_gross, _ = newey_west_tstat(gross)
    net = (res[f"ret_{port}"] - res["mkt_fwd"] - res[f"TO_{port}"] * res["cost_rt"]).dropna()
    t_net = None
    if not net.empty:
        _, t_n, _ = newey_west_tstat(net)
        t_net = round(float(t_n), 3)
    return {"port": port, "status": "OK",
            "nw_t": round(float(t_gross), 3), "caliber": "gross_excess",
            "nw_t_net": t_net, "n": int(len(gross)),
            "gross_mean_pct": round(float(gross.mean()) * 100, 4),
            "net_mean_pct": round(float(net.mean()) * 100, 4) if not net.empty else None}


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any],
                        *, t_drop_alert: float = 0.5) -> list[dict[str, Any]]:
    """当前体检 vs 冻结基线,产出需人工复审的发现。

    判据(全部复用既有门,唯一新增的 t_drop_alert 是"材料性下降"的呈现阈值,不是准入门,
    默认 0.5 —— 说明:它只决定"要不要提醒你看一眼",不改变任何因子的去留):
    - 因子由 PASS 变 FAIL → DEGRADED(准入证据不再成立,必须人工复审)
    - 因子仍 PASS 但 NW t 较基线下降 ≥ t_drop_alert → DRIFT(观察)
    - 组合 NW t 跌破 T_ADMIT/T_FOLD 或较基线下降 ≥ t_drop_alert → 相应等级
    """
    from scripts.factor_tearsheet import T_ADMIT, T_FOLD

    out: list[dict[str, Any]] = []
    base_factors = {f["factor"]: f for f in baseline.get("factors", [])}
    for cur in current.get("factors", []):
        name = cur["factor"]
        base = base_factors.get(name)
        if cur["status"] == "NO_DATA":
            out.append({"level": "DEGRADED", "target": name, "issue": "NO_DATA",
                        "detail": "; ".join(cur.get("reasons", [])) or "读数缺失,无法验证准入证据"})
            continue
        if cur["status"] == "FAIL":
            out.append({"level": "DEGRADED", "target": name, "issue": "GATE_FAIL",
                        "detail": f"不再过五门: {'; '.join(cur['reasons'])}"
                                  f"(当初入分的证据不再成立——人工复审,勿自动改 composite)"})
            continue
        if base and base.get("nw_t") is not None:
            # **按 |t| 比较**:IVOL 等负向因子的 t 为负,直接做差会把 -17→-10 这种
            # 显著性大幅退化算成"上升"而漏报(codex P1)。门本身就是 |t|>T_ADMIT,
            # 漂移也必须同口径看绝对显著性。
            # 变号优先于强度下降判定:方向翻转比 |t| 变小严重得多(因子语义已变),
            # 若先命中 DRIFT 分支就会把它降级成观察项
            if float(base["nw_t"]) * float(cur["nw_t"]) < 0:
                out.append({"level": "DEGRADED", "target": name, "issue": "T_SIGN_FLIP",
                            "detail": f"NW t 由 {base['nw_t']} 变号为 {cur['nw_t']}"
                                      "——方向翻转,因子语义已变,必须人工复审"})
            else:
                drop = abs(float(base["nw_t"])) - abs(float(cur["nw_t"]))
                if drop >= t_drop_alert:
                    out.append({"level": "DRIFT", "target": name, "issue": "T_DROP",
                                "detail": f"NW t {base['nw_t']} → {cur['nw_t']}"
                                          f"(|t| 降 {drop:.2f});仍过 T_ADMIT={T_ADMIT},属观察项"})

    cur_c = current.get("composite") or {}
    base_c = baseline.get("composite") or {}
    if cur_c.get("status") == "NO_DATA":
        out.append({"level": "DEGRADED", "target": "composite", "issue": "NO_DATA",
                    "detail": cur_c.get("note", "组合读数缺失")})
    elif cur_c.get("nw_t") is not None:
        t = float(cur_c["nw_t"])
        # 组合超额的**方向有经济含义**(负超额=跑输基准),故这里不取绝对值:
        # t 本身低于 T_FOLD(含变负)都要报——与因子腿按 |t| 比较是两种口径,勿混
        if t < T_FOLD:
            out.append({"level": "DEGRADED", "target": "composite", "issue": "T_BELOW_FOLD",
                        "detail": f"组合毛超额 NW t={t} < {T_FOLD}——人工复审(是风格逆风、"
                                  f"成本口径、还是失效?复审前不得自动改因子)"})
        elif base_c.get("nw_t") is not None:
            drop = float(base_c["nw_t"]) - t
            if drop >= t_drop_alert:
                out.append({"level": "DRIFT", "target": "composite", "issue": "T_DROP",
                            "detail": f"组合毛超额 NW t {base_c['nw_t']} → {t}(降 {drop:.2f})"})
    return out


def build_report(detail_rows: Sequence[dict[str, Any]],
                 composite_rows: Sequence[dict[str, Any]] | None,
                 *, as_of: str | None = None) -> dict[str, Any]:
    """跑一次完整体检(不含基线比对)。"""
    factors = [factor_gate_status(detail_rows, f) for f in PRODUCTION_FACTORS]
    comp = composite_nw_t(composite_rows) if composite_rows else {
        "port": "PROD", "status": "NO_DATA", "nw_t": None, "n": 0,
        "note": "未提供 composite_backtest 读数"}
    dates = [str(r.get("date")) for r in detail_rows if r.get("date")]
    return {
        "schema_version": "gate_health.v1",
        "as_of": as_of,
        "sample": {"n": len(detail_rows),
                   "first": min(dates) if dates else None,
                   "last": max(dates) if dates else None},
        "factors": factors,
        "composite": comp,
        "all_gates_pass": all(f["status"] == "PASS" for f in factors),
    }
