"""Cherry Studio stdio MCP server for ashare-gauntlet.

Run from any directory:
    E:\\zwyq\\.venv\\Scripts\\python.exe E:\\zwyq\\scripts\\mcp_server.py

Stdout belongs to the MCP wire.  Do not add import-time prints.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct file execution puts scripts/ rather than the repository on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

from ashare_gauntlet import mcp_service as service

mcp = FastMCP(
    "ashare-gauntlet",
    instructions=(
        "A股量化研究、机器状态查询与个人账户风险辅助；本服务器严格只读，"
        "不刷新数据、不生成产物、不执行交易。最新机器决策是正式状态唯一真相源，"
        "因子、腾讯报价和触发带只能补充证据，不能把WAIT提升为BUY。"
        "没有已验证的结构化入场价格时，用户动作必须保持WAIT，精确价格为null。"
        "条件单缺失、未核验或非法不是荐股、配股或落账门禁，不得据此停止推荐。"
        "推荐不等于成交；未经用户确认不得修改持仓或声称已下单。"
    ),
)


@mcp.tool()
def healthcheck() -> dict:
    """检查项目、虚拟环境、动态数据与配置是否就绪；绝不返回API密钥。"""
    return service.healthcheck()


@mcp.tool()
def get_account_snapshot() -> dict:
    """读取持仓单一真相源并计算总资产、行业权重与短线席位。条件单仅解析展示，不是荐股或执行门禁。"""
    return service.account_snapshot()


@mcp.tool()
def get_strategy_context() -> dict:
    """读取交易政策、基金行业排除、fact-check和盘中触发带。"""
    return service.strategy_context()


@mcp.tool()
def get_latest_decisions(states: list[str] | None = None, offset: int = 0,
                         limit: int = 50, compact: bool = True,
                         summary_only: bool = False) -> dict:
    """分页读取机器决策；先summary_only，再按BUY/EXIT状态取明细。"""
    return service.latest_decisions(states=states, offset=offset, limit=limit,
                                    compact=compact, summary_only=summary_only)


@mcp.tool()
def get_factor_candidates(deciles: list[int] | None = None, offset: int = 0,
                          limit: int = 20, compact: bool = True) -> dict:
    """分页读取因子候选；默认compact，可按deciles=[10]过滤。"""
    return service.latest_factor_rows(deciles=deciles, offset=offset, limit=limit,
                                      compact=compact)


@mcp.tool()
def get_intraday_quotes(ts_codes: list[str]) -> dict:
    """获取腾讯最新可用行情快照；仅会话和日期对齐时标为盘中。"""
    return service.intraday_quotes(ts_codes)


@mcp.tool()
def get_stock_brief(ts_code: str, include_intraday: bool = True) -> dict:
    """汇总单股机器决策及证据；触发带仅为advisory，不改变正式状态。"""
    return service.stock_brief(ts_code, include_intraday=include_intraday)


@mcp.tool()
def check_governance(ts_codes: list[str], force_refresh: bool = False) -> dict:
    """只读本地质押与审计缓存；未覆盖不等于无风险，force_refresh已禁用。"""
    return service.governance_check(ts_codes, force=force_refresh)


@mcp.prompt()
def daily_recommendation_prompt(max_recommendations: int = 3) -> str:
    """生成每日实盘账户荐股任务模板(账户规模由 get_account_snapshot 提供)。"""
    if not 1 <= max_recommendations <= 3:
        raise ValueError("max_recommendations must be 1..3")
    return f"""先调用 healthcheck。只有 recommendation_readiness.ready=true 才能正式荐股；
ok=true 只表示服务可用。若 blocked，停止荐股并逐项报告 blockers。
然后调用 get_account_snapshot、get_strategy_context、get_latest_decisions(summary_only=true)，
再调用 get_latest_decisions(states=[\"BUY\",\"EXIT\"], compact=true, limit=50)。
仅需解释等待项时再分页调用 get_factor_candidates(deciles=[10], compact=true)。
最新 decision 的 state 是正式状态唯一真相源；
因子、fact-check、腾讯现价和 trigger band 只能解释或补充，绝不能把 WAIT 提升为 BUY。
必要时调用 get_stock_brief。最多展示 {max_recommendations} 只，并逐只给出 machine_state、
user_action、reason_codes、actionable、股数、风险、放弃条件、EOD日期和盘中时间。
精确买入区间、委托价、止损价只能原样引用 decision 中已验证且非空的结构化字段；
max_entry_price 或相关证据缺失时，user_action 必须为 WAIT，对应价格字段写 null，
不得用 trigger band、盘中现价、“约”或“附近”补造。trigger band 只能标为 advisory。
条件单缺失、未核验或非法不是 blockers，不得据此停止荐股、配股或落账，也不得伪报已核验。
短线席位占用时不得推荐第二只短线股。推荐不等于成交。若无可执行 BUY，明确说
“今日无合格BUY，不为凑名单而推荐”。"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
