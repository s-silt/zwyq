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


def tushare_pro():
    """标准 tushare 装配:``.env.local`` 权威覆盖 → ``make_pro_api(TOKEN, HTTP_URL)``。

    TUSHARE_TOKEN / TUSHARE_HTTP_URL 任一缺失即 KeyError fail-loud(与散落各
    脚本的历史语义一致,不静默降级到官方网关)。tushare_source 延迟导入,
    使只读路径常量的轻消费方(如盘中哨兵)不背 tushare 依赖。
    """
    load_env_local()
    from ashare_gauntlet.data import tushare_source

    return tushare_source.make_pro_api(
        os.environ["TUSHARE_TOKEN"], os.environ["TUSHARE_HTTP_URL"]
    )
