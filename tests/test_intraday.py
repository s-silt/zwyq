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
    assert q["600875.SH"]["quote_as_of"] == "2026-07-08T15:00:00+08:00"
    assert q["600875.SH"]["trade_date"] == "20260708"
    assert q["600875.SH"]["timestamp_status"] == "valid"


def test_parse_tencent_quote_multiple_and_junk_lines():
    two = SAMPLE + '\nv_pv_none="";\n' + SAMPLE.replace("sh600875", "sz000589").replace(
        "600875", "000589").replace("东方电气", "贵州轮胎")
    q = parse_tencent_quote(two)
    assert set(q) == {"600875.SH", "000589.SZ"}


def test_parse_tencent_quote_rejects_symbol_code_mismatch():
    mismatched = SAMPLE.replace("v_sh600875", "v_sh600001", 1)
    with pytest.raises(ValueError, match="无有效行"):
        parse_tencent_quote(mismatched)


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


# ---------- 止盈提醒 + 定时任务去重(2026-07-10 用户批准) ----------

def test_alert_profit_when_above_tp():
    # 长线 +25% 减半锁利(既有约定):现价≥tp → PROFIT;止损类警报优先级更高
    assert alert_level(last=12.6, stop=10.0, warn_dist=0.03, tp=12.5) == "PROFIT"
    assert alert_level(last=12.4, stop=10.0, warn_dist=0.03, tp=12.5) == "OK"
    assert alert_level(last=10.2, stop=10.0, warn_dist=0.03, tp=12.5) == "NEAR"  # 止损优先


def test_sentinel_state_namespaced_keys_no_collision():
    # 设计审查[高]:众兴同时在持仓与观察名单,ts_code 裸键互相覆盖 → pos:/watch: 命名空间
    from ashare_gauntlet.intraday import sentinel_delta
    cur = {"pos:002772.SZ": "NEAR", "watch:002772.SZ": "BAND"}
    report, cleared, state = sentinel_delta({}, cur, "20260710")
    assert set(report) == {"pos:002772.SZ", "watch:002772.SZ"}   # 两条独立警报都报


def test_sentinel_breach_latch_no_oscillation_spam():
    # 设计审查[高]:BREACH→NEAR→BREACH 震荡不得反复轰炸——当日已报最高级封存(latch)
    from ashare_gauntlet.intraday import sentinel_delta
    _, _, st = sentinel_delta({}, {"pos:a": "BREACH"}, "20260710")            # 首报 BREACH
    r2, _, st = sentinel_delta(st, {"pos:a": "NEAR"}, "20260710")             # 回到 NEAR:静默
    assert r2 == []
    r3, _, st = sentinel_delta(st, {"pos:a": "BREACH"}, "20260710")           # 再破线:仍静默(已 latch)
    assert r3 == []


def test_sentinel_breach_clear_reported_once():
    # 设计审查[中]:BREACH 解除要报一次(信息缺口),但只报一次
    from ashare_gauntlet.intraday import sentinel_delta
    _, _, st = sentinel_delta({}, {"pos:a": "BREACH"}, "20260710")
    r, cleared, st = sentinel_delta(st, {"pos:a": "OK"}, "20260710")
    assert cleared == ["pos:a"]
    _, cleared2, st = sentinel_delta(st, {"pos:a": "OK"}, "20260710")
    assert cleared2 == []                                                     # 不重复报解除


def test_sentinel_new_day_resets_state():
    # 跨交易日状态重置:昨日 latch 不带入今天(PROFIT/BREACH 每日至多一报)
    from ashare_gauntlet.intraday import sentinel_delta
    _, _, st = sentinel_delta({}, {"pos:a": "PROFIT"}, "20260710")
    r, _, st = sentinel_delta(st, {"pos:a": "PROFIT"}, "20260713")            # 新交易日
    assert r == ["pos:a"]


def test_sentinel_fingerprint_change_resets_key():
    # 设计审查[中]:持仓指纹(股数@成本@止损)变化 → 该键状态重置(卖出再买不继承旧 latch)
    from ashare_gauntlet.intraday import sentinel_delta
    _, _, st = sentinel_delta({}, {"pos:a": "BREACH"}, "20260710", fps={"pos:a": "800@10.5@10"})
    r, _, st = sentinel_delta(st, {"pos:a": "BREACH"}, "20260710", fps={"pos:a": "500@9.8@9.2"})
    assert r == ["pos:a"]                                                     # 新仓,重新报


# ---------- 实现审查修复(Codex 哨兵批):解除语义/重破线/合成键 ----------

def test_sentinel_breach_to_profit_also_clears():
    # P1:BREACH→PROFIT(如止损线下方反弹直接冲过止盈线)也算解除,不只 OK
    from ashare_gauntlet.intraday import sentinel_delta
    _, _, st = sentinel_delta({}, {"pos:a": "BREACH"}, "20260710")
    _, cleared, st = sentinel_delta(st, {"pos:a": "PROFIT"}, "20260710")
    assert cleared == ["pos:a"]


def test_sentinel_rebreach_after_clear_reports_again():
    # P1:解除后同日再破线必须重报——用户最后看到的是"安全",再破线静默=危险漏报
    from ashare_gauntlet.intraday import sentinel_delta
    _, _, st = sentinel_delta({}, {"pos:a": "BREACH"}, "20260710")
    _, _, st = sentinel_delta(st, {"pos:a": "OK"}, "20260710")        # 解除
    r, _, st = sentinel_delta(st, {"pos:a": "BREACH"}, "20260710")    # 再破线
    assert r == ["pos:a"]
    r2, _, st = sentinel_delta(st, {"pos:a": "BREACH"}, "20260710")   # 再 latch,不轰炸
    assert r2 == []


def test_sentinel_synthetic_meta_keys_report_once_daily():
    # P1:持仓陈旧/缺行情作为合成键走状态机——每日一报,不随价格警报静默而丢失
    from ashare_gauntlet.intraday import sentinel_delta
    cur = {"meta:holdings_stale": "NEAR", "miss:600999.SH": "NEAR"}
    r, _, st = sentinel_delta({}, cur, "20260710")
    assert set(r) == set(cur)
    r2, _, st = sentinel_delta(st, cur, "20260710")
    assert r2 == []                                                    # 同日不重复
