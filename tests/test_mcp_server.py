import asyncio
import json

import pytest

from scripts.mcp_maintenance_server import mcp as maintenance_mcp
from scripts.mcp_server import daily_recommendation_prompt, mcp


def test_mcp_tools_keep_single_text_payload_and_no_output_schema(tmp_path, monkeypatch) -> None:
    from ashare_gauntlet import mcp_service as svc

    monkeypatch.setattr(svc, "latest_decisions", lambda **kwargs: {
        "as_of": "20260807",
        "summary": {"state_counts": {"BUY": 0, "WAIT": 170, "HOLD": 6, "EXIT": 0}},
        "page": {"returned": 0, "has_more": False},
        "decisions": [],
    })
    tools = asyncio.run(mcp.list_tools())
    decisions_tool = next(tool for tool in tools if tool.name == "get_latest_decisions")
    assert decisions_tool.outputSchema is None

    result = asyncio.run(mcp.call_tool("get_latest_decisions", {"summary_only": True}))
    assert isinstance(result, list) and len(result) == 1
    assert getattr(result[0], "type", None) == "text"
    payload = json.loads(result[0].text)
    assert payload["decisions"] == []
    assert len(result[0].text.encode("utf-8")) < 32768


def test_read_and_maintenance_servers_have_disjoint_tool_surfaces() -> None:
    read_tools = {tool.name for tool in asyncio.run(mcp.list_tools())}
    maintenance_tools = {
        tool.name for tool in asyncio.run(maintenance_mcp.list_tools())
    }
    assert read_tools == {
        "healthcheck", "get_account_snapshot", "get_strategy_context",
        "get_latest_decisions", "get_factor_candidates", "get_intraday_quotes",
        "get_stock_brief", "check_governance",
    }
    assert maintenance_tools == {
        "refresh_eod", "refresh_stock_financials", "generate_factor_snapshot",
        "generate_machine_decisions", "generate_personal_aggressive_view",
        "calculate_market_temperature",
    }
    assert read_tools.isdisjoint(maintenance_tools)
    assert "generate_daily_analysis" not in maintenance_tools


def test_mcp_invalid_decision_state_is_tool_error(monkeypatch) -> None:
    from ashare_gauntlet import mcp_service as svc

    def fail(**kwargs):
        raise ValueError("states must contain values")

    monkeypatch.setattr(svc, "latest_decisions", fail)
    with pytest.raises(Exception, match="states must contain values"):
        asyncio.run(mcp.call_tool("get_latest_decisions", {"states": ["BAD"]}))


def test_daily_prompt_uses_readiness_and_bounded_decision_calls() -> None:
    prompt = daily_recommendation_prompt()
    assert "recommendation_readiness.ready=true" in prompt
    assert "summary_only=true" in prompt
    assert 'states=["BUY","EXIT"]' in prompt
    assert "不能把 WAIT 提升为 BUY" in prompt


def test_check_governance_mcp_returns_structured_local_payload(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from ashare_gauntlet import mcp_service as svc

    pledge = pd.DataFrame([{
        "ann_date": "20260731", "holder_name": "股东甲",
        "pledged_amount": 10.0, "holding_amount": 100.0, "h_total_ratio": 5.0,
    }])
    audit = pd.DataFrame([{
        "end_date": "20251231", "audit_result": "标准无保留意见",
    }])
    for endpoint, frame in (("pledge_detail", pledge), ("fina_audit", audit)):
        path = tmp_path / "data/cache" / endpoint / "600875.SH.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    monkeypatch.setattr(svc, "project_root", lambda: tmp_path)
    monkeypatch.setattr(svc, "run_module", lambda *args, **kwargs: pytest.fail("subprocess used"))

    result = asyncio.run(mcp.call_tool(
        "check_governance", {"ts_codes": ["600875.SH"], "force_refresh": False},
    ))
    assert isinstance(result, list) and len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["ok"] is True
    assert payload["network_access"] is False
    assert payload["source"] == "local_parquet"
    assert not {"stdout", "stderr", "returncode", "module"} & payload.keys()


def test_check_governance_mcp_rejects_force_refresh() -> None:
    with pytest.raises(Exception, match="refresh_stock_financials"):
        asyncio.run(mcp.call_tool(
            "check_governance", {"ts_codes": ["600875.SH"], "force_refresh": True},
        ))
