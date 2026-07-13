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
    assert config.PROFILE_PATH == "data/profile.json"


def test_tushare_pro_fails_loud_without_token(tmp_path, monkeypatch):
    """无 .env.local 且环境无 token → KeyError(与散落各脚本的历史语义一致,不静默降级)。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_HTTP_URL", raising=False)
    from ashare_gauntlet import config

    with pytest.raises(KeyError):
        config.tushare_pro()


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
