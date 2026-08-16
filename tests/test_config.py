"""config 集中层:路径口径钉死 + tushare_pro 装配语义(fail-loud/.env.local 权威)。

这些字符串此前散落在 20+ 个 scripts 里各自定义(外部评审 P2 指出的重复);
钉死在此防止收敛后有人改常量却漏改消费方的隐式约定(缓存/产物目录是磁盘契约)。
"""
from __future__ import annotations

import pytest


def test_paths_pin_historical_strings():
    from ashare_gauntlet import config

    assert config.CACHE_DIR == "data/cache"
    assert config.HOLDSCORE_DIR == "data/holdscore"
    assert config.SURVIVORS_DIR == "data/survivors"
    assert config.CARDS_DIR == "data/cards"
    assert config.PANELS_DIR == "data/panels"
    assert config.CARDS_SVG_DIR == "data/cards_svg"
    assert config.BYSTOCK_DIR == "data/bystock"
    assert config.HOLDINGS_PATH == "data/holdings.json"
    assert config.TRIGGER_BANDS_PATH == "data/trigger_bands.json"
    assert config.WATCHLIST_PATH == "data/watchlist.json"
    assert config.TRADE_JOURNAL_PATH == "data/trade_journal.json"
    assert config.INTRADAY_STATE_PATH == "data/intraday_alert_state.json"
    assert config.ACCOUNT_STATE_DIR == "data/account_state"
    assert config.PROFILE_PATH == "data/profile.json"


def test_tushare_pro_fails_loud_without_token(tmp_path, monkeypatch):
    """无 .env.local 且环境无 token → 明确失败，不静默创建匿名客户端。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_HTTP_URL", raising=False)
    from ashare_gauntlet import config

    with pytest.raises(RuntimeError, match="TUSHARE_TOKEN"):
        config.tushare_pro()


def test_tushare_pro_allows_token_without_http_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "tk")
    monkeypatch.delenv("TUSHARE_HTTP_URL", raising=False)
    from ashare_gauntlet import config
    from ashare_gauntlet.data import tushare_source

    captured: dict = {}

    def fake(token: str, http_url: str | None = None, timeout: int = 120):
        captured.update(token=token, url=http_url)
        return "PRO"

    monkeypatch.setattr(tushare_source, "make_pro_api", fake)
    assert config.tushare_pro() == "PRO"
    assert captured == {"token": "tk", "url": None}


def test_tushare_pro_strict_env_admits_probe_keys(tmp_path, monkeypatch):
    """strict 模式必须放行 factcheck_probe 的密钥(走 tushare_pro 真入口,锁接线):
    mcp_service 以 strict_env=True 加载 .env.local,白名单漏键会把 MCP 掀翻。"""
    monkeypatch.chdir(tmp_path)
    keys = ("TUSHARE_TOKEN", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
            "DEEPSEEK_MODEL", "KAGI_API_KEY")
    for key in keys:   # 先经 monkeypatch 登记,teardown 还原——load_env_local 会写全局 env
        monkeypatch.setenv(key, "sentinel")
    (tmp_path / ".env.local").write_text(
        "TUSHARE_TOKEN=tk\nDEEPSEEK_API_KEY=d\nDEEPSEEK_BASE_URL=https://gw/v1\n"
        "DEEPSEEK_MODEL=m\nKAGI_API_KEY=k\n", encoding="utf-8"
    )
    from ashare_gauntlet import config
    from ashare_gauntlet.data import tushare_source

    monkeypatch.setattr(tushare_source, "make_pro_api", lambda *a, **k: "PRO")
    assert config.tushare_pro(strict_env=True) == "PRO"


def test_tushare_pro_assembles_from_env_local(tmp_path, monkeypatch):
    """.env.local 权威覆盖进程环境后,按 (TOKEN, HTTP_URL) 装配 make_pro_api。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text(
        "TUSHARE_TOKEN=tk\nTUSHARE_HTTP_URL=http://x\n", encoding="utf-8"
    )
    from ashare_gauntlet import config
    from ashare_gauntlet.data import tushare_source

    captured: dict = {}

    def fake(token: str, http_url: str, timeout: int = 120):
        captured.update(token=token, url=http_url)
        return "PRO"

    monkeypatch.setattr(tushare_source, "make_pro_api", fake)
    assert config.tushare_pro() == "PRO"
    assert captured == {"token": "tk", "url": "http://x"}
