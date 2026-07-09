"""盘中价格哨兵(stock-sdk 评估后吸纳的唯一能力:盘中实时价,自实现零新依赖)。

数据源=腾讯行情公开接口(qt.gtimg.cn,GBK 编码,延迟秒-分钟级);**口径隔离**:
盘中价只用于提醒,绝不写入研究缓存(tushare EOD 口径不混,守数据源纯净审计线)。
警报阈值是操作性提醒参数(CLI 可调),非评分常数——零 magic number 公约管评分层。
"""
import math

import pytest

from ashare_gauntlet.intraday import alert_level, parse_tencent_quote, tencent_symbol

# 腾讯接口真实返回格式样例(GBK 解码后;字段以 ~ 分隔,[1]名 [3]现价 [4]昨收 [5]今开 [32]涨跌幅)
SAMPLE = ('v_sh600875="1~东方电气~600875~27.71~28.40~28.30~123456~1~2~27.70~1~'
          '27.71~2~27.72~3~27.73~4~27.74~5~27.69~1~27.68~2~27.67~3~27.66~4~27.65~5~'
          '~20260708150000~-0.69~-2.43~28.35~27.55~27.71/123456/342000000~123456~34200~'
          '0.79~15.2~~28.35~27.55~2.82~432.5~438.1~1.2~31.24~25.56~1.1~~~~~";')


def test_parse_tencent_quote_fields():
    q = parse_tencent_quote(SAMPLE)
    assert q["600875.SH"]["name"] == "东方电气"
    assert q["600875.SH"]["last"] == pytest.approx(27.71)
    assert q["600875.SH"]["prev_close"] == pytest.approx(28.40)
    assert q["600875.SH"]["pct"] == pytest.approx(-2.43)


def test_parse_tencent_quote_multiple_and_junk_lines():
    two = SAMPLE + '\nv_pv_none="";\n' + SAMPLE.replace("sh600875", "sz000589").replace(
        "600875", "000589").replace("东方电气", "贵州轮胎")
    q = parse_tencent_quote(two)
    assert set(q) == {"600875.SH", "000589.SZ"}


def test_parse_tencent_quote_empty_fails_loud():
    # 空响应=接口异常,拒绝静默返回空 dict(哨兵沉默比误报危险)
    with pytest.raises(ValueError):
        parse_tencent_quote("")


def test_tencent_symbol_roundtrip():
    assert tencent_symbol("600875.SH") == "sh600875"
    assert tencent_symbol("000589.SZ") == "sz000589"
    with pytest.raises(ValueError):
        tencent_symbol("600875")        # 非 ts_code 格式 fail-loud,防静默查错票


# ---------- 警报分级:BREACH(已破止损)> NEAR(逼近)> BAND(触发带命中)> OK ----------

def test_alert_breach_when_at_or_below_stop():
    assert alert_level(last=9.99, stop=10.0, warn_dist=0.03) == "BREACH"
    assert alert_level(last=10.0, stop=10.0, warn_dist=0.03) == "BREACH"


def test_alert_near_within_warn_distance():
    assert alert_level(last=10.25, stop=10.0, warn_dist=0.03) == "NEAR"   # 距 2.5%


def test_alert_ok_when_far():
    assert alert_level(last=11.0, stop=10.0, warn_dist=0.03) == "OK"


def test_alert_band_hit():
    # 触发带(观察名单):last 落入 [band_low, band_high] → BAND
    assert alert_level(last=12.8, stop=None, warn_dist=0.03,
                       band=(12.5, 13.0)) == "BAND"
    assert alert_level(last=13.2, stop=None, warn_dist=0.03,
                       band=(12.5, 13.0)) == "OK"


def test_alert_no_stop_no_band_is_ok():
    assert alert_level(last=5.0, stop=None, warn_dist=0.03) == "OK"


# ---------- stock-sdk 对比审查吸纳(2026-07-09):协议健壮性加固 ----------

def test_parse_semicolon_separated_single_line():
    # 腾讯响应可能单行多记录(分号分隔,无换行)——splitlines 解析会串字段;
    # 改正则提取后两种物理格式都要通(stock-sdk parser.ts 同款语义)
    one_line = SAMPLE.rstrip() + 'v_sz000589="1~贵州轮胎~000589~4.27~4.30~4.28' + "~x" * 26 + '~-0.70~~~";'
    q = parse_tencent_quote(one_line)
    assert set(q) == {"600875.SH", "000589.SZ"}


def test_parse_ignores_pv_none_match():
    # v_pv_none_match 空壳行(请求了不存在的符号时腾讯返回)不得混入结果
    q = parse_tencent_quote(SAMPLE + '\nv_pv_none_match="1";')
    assert set(q) == {"600875.SH"}


def test_fetch_chunking_merges_batches():
    # 批量上限切片(stock-sdk MAX_BATCH_SIZE=500;URL 超长会脆断)——注入 fetcher 验证合并
    from ashare_gauntlet.intraday import chunked
    codes = [f"c{i}" for i in range(1201)]
    batches = list(chunked(codes, 500))
    assert [len(b) for b in batches] == [500, 500, 201]
    assert sum(batches, []) == codes
