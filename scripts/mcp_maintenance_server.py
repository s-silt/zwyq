r"""Opt-in maintenance-only stdio MCP server for ashare-gauntlet.

Run explicitly after configuring per-call approval:
    E:\zwyq\.venv\Scripts\python.exe E:\zwyq\scripts\mcp_maintenance_server.py

Every tool may access the network or write reproducible cache/artifact files.
This server never edits manual account state and exposes no recommendation prompt.
Stdout belongs to the MCP wire. Do not add import-time prints.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

from ashare_gauntlet import mcp_service as service

mcp = FastMCP(
    "ashare-gauntlet-maintenance",
    instructions=(
        "A股量化项目的显式维护服务器。每个工具都可能联网或写入可再生缓存/产物，"
        "必须逐次人工确认；不修改 holdings、trading_policy、profile、fact-check、"
        "trigger bands 或交易流水，不执行下单。各分析阶段彼此独立，不自动串联。"
    ),
)


@mcp.tool()
async def refresh_eod(start_date: str, end_date: str) -> dict:
    """严格确保区间内每个开市日四个核心 EOD 端点完整。"""
    return await asyncio.to_thread(service.refresh_eod, start_date, end_date)


@mcp.tool()
async def refresh_stock_financials(ts_codes: list[str]) -> dict:
    """联网刷新至多 10 只股票的财务、治理和事件缓存。"""
    return await asyncio.to_thread(service.refresh_financials, ts_codes)


@mcp.tool()
async def generate_factor_snapshot(top: int = 20) -> dict:
    """仅基于已有 EOD 生成生产因子快照。"""
    return await asyncio.to_thread(service.generate_factor_snapshot, top=top)


@mcp.tool()
async def generate_machine_decisions() -> dict:
    """仅基于现有因子快照和人工账户状态生成四态机器决策。"""
    return await asyncio.to_thread(service.generate_machine_decisions)


@mcp.tool()
async def generate_personal_aggressive_view(top: int = 20) -> dict:
    """仅生成个人进攻视图；该产物不是正式机器决策真相源。"""
    return await asyncio.to_thread(service.generate_personal_aggressive_view, top=top)


@mcp.tool()
async def calculate_market_temperature() -> dict:
    """仅生成市场温度产物。"""
    return await asyncio.to_thread(service.calculate_market_temperature)


if __name__ == "__main__":
    mcp.run(transport="stdio")
