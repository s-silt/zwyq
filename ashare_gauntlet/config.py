"""集中配置层:数据/产物路径 + tushare 装配的唯一事实源(外部评审 P2 收敛)。

此前 ``CACHE = "data/cache"``、各产物目录、以及"``load_env_local()`` +
``make_pro_api(TOKEN, HTTP_URL)``"三行装配在 20+ 个 scripts 里各自复制。
收敛在此;scripts 以别名导入(``from ashare_gauntlet.config import CACHE_DIR
as CACHE``)保持原有代码体不变。

所有路径为**相对项目根**的字符串(scripts 均以仓库根为 cwd 运行,历史约定);
不做 Path 化/绝对化,避免改变既有消费方的拼接语义。
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from ashare_gauntlet.data.env import load_env_local

# —— 数据缓存(tushare parquet 镜像,可再生,勿手改)——
CACHE_DIR = "data/cache"
# —— 研究/呈现产物目录(均可再生)——
HOLDSCORE_DIR = "data/holdscore"      # 因子排名/回测明细/pick_track
SURVIVORS_DIR = "data/survivors"      # 四关漏斗前三关输出
CARDS_DIR = "data/cards"              # 全市场建档 record
PANELS_DIR = "data/panels"            # 面板
CARDS_SVG_DIR = "data/cards_svg"      # 单股 SVG 卡
BYSTOCK_DIR = "data/bystock"          # 按股票原始拉取(pull_symbols)
# —— 手工维护的个人数据 / 运行时状态 ——
HOLDINGS_PATH = "data/holdings.json"              # 实盘持仓单一真相源
TRIGGER_BANDS_PATH = "data/trigger_bands.json"    # 观察名单触发带(哨兵消费)
WATCHLIST_PATH = "data/watchlist.json"            # 6/19 种子名单(cards/tech_report 消费)
TRADE_JOURNAL_PATH = "data/trade_journal.json"    # 交易流水
INTRADAY_STATE_PATH = "data/intraday_alert_state.json"  # 哨兵去重状态(可删)
ACCOUNT_STATE_DIR = "data/account_state"          # EOD 账户估值快照(运行态,可再生)
PROFILE_PATH = "data/profile.json"                # 个人投资约束 profile(非研究结论)


def tushare_pro(*, env_path: str | os.PathLike[str] = ".env.local",
                strict_env: bool = False):
    """标准 Tushare 装配:token 必需，HTTP URL 为可选镜像覆盖。

    ``.env.local`` 仍是权威覆盖源。未配置或留空 ``TUSHARE_HTTP_URL`` 时
    保留 SDK 官方端点与调用方代理；显式镜像由 ``make_pro_api`` 处理直连。
    tushare_source 延迟导入，避免路径常量消费方承担 Tushare 依赖。
    """
    allowed = {"TUSHARE_TOKEN", "TUSHARE_HTTP_URL"} if strict_env else None
    load_env_local(env_path, allowed_keys=allowed)
    from ashare_gauntlet.data import tushare_source

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is required")
    http_url = (os.environ.get("TUSHARE_HTTP_URL") or "").strip() or None
    if strict_env and http_url:
        parsed = urlparse(http_url)
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and parsed.hostname not in local_hosts:
            raise RuntimeError("TUSHARE_HTTP_URL must use HTTPS in MCP mode")
    return tushare_source.make_pro_api(token, http_url)
