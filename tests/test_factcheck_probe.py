"""factcheck_probe:证据抓取管线的契约测试(无网络;HTTP 全部注入假对象)。

守住三条底线:密钥缺失 fail-loud、空材料拒绝抽取(禁模型记忆)、
候选选择只收"仅差 fact-check"的绿灯 WAIT 且小桶优先。
"""
from __future__ import annotations

import json

import pytest

from scripts import factcheck_probe as fp


class _Resp:
    def __init__(self, status_code: int = 200, payload=None, text: str = "",
                 content: bytes = b"", headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._payload


class _Http:
    """记录调用参数的假 requests。"""

    def __init__(self, get_responses: list[_Resp] | None = None,
                 post_responses: list[_Resp] | None = None):
        self.get_calls: list[dict] = []
        self.post_calls: list[dict] = []
        self._gets = list(get_responses or [])
        self._posts = list(post_responses or [])

    def get(self, url, **kwargs):
        self.get_calls.append({"url": url, **kwargs})
        return self._gets.pop(0)

    def post(self, url, **kwargs):
        self.post_calls.append({"url": url, **kwargs})
        return self._posts.pop(0)


def _deepseek_resp(content: dict) -> _Resp:
    return _Resp(payload={"choices": [{"message": {"content": json.dumps(content)}}]})


def test_require_keys_fail_loud_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAGI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(fp.FactcheckProbeError, match="DEEPSEEK_API_KEY"):
        fp.require_keys()


def test_require_keys_kagi_optional(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text("DEEPSEEK_API_KEY=d1\n", encoding="utf-8")
    monkeypatch.delenv("KAGI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert fp.require_keys() == ("", "d1")


def test_probe_one_without_kagi_key_fail_loud():
    with pytest.raises(fp.FactcheckProbeError, match="--sources"):
        fp.probe_one(_Http(), "", "d", "X", "000001.SZ")


def test_probe_from_urls_extracts_without_search():
    content = {"q1_net_profit_yi": 1.03, "q1_yoy_pct": None,
               "q1_sources": ["https://a", "https://b"],
               "risks": [{"claim": "减持", "source_url": "https://b"}],
               "not_found": [], "contradictions": []}
    http = _Http(post_responses=[_deepseek_resp(content)])
    record = fp.probe_from_urls(http, "d", "富春环保", "002479.SZ",
                                ["https://a", "https://b"],
                                fetcher=lambda _h, u: f"{u} 的正文")
    assert record["confirmed"] is True
    assert http.get_calls == []          # sources 模式绝不发检索请求
    body = http.post_calls[0]["json"]
    assert "https://a 的正文" in body["messages"][1]["content"]


def test_require_keys_loads_env_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text(
        "KAGI_API_KEY=k1\nDEEPSEEK_API_KEY=d1\n", encoding="utf-8")
    monkeypatch.delenv("KAGI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert fp.require_keys() == ("k1", "d1")


def test_strict_env_keys_cover_probe_keys():
    # mcp_service 以 strict 模式加载 .env.local;新密钥不进白名单会把 MCP 掀翻
    from ashare_gauntlet.config import STRICT_ENV_KEYS

    assert {"KAGI_API_KEY", "DEEPSEEK_API_KEY",
            "TUSHARE_TOKEN", "TUSHARE_HTTP_URL"} <= STRICT_ENV_KEYS


def test_deepseek_endpoint_env_override(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    assert fp._deepseek_endpoint() == (fp.DEEPSEEK_CHAT_URL, fp.DEEPSEEK_MODEL)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gw.example/v1/")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v3")
    assert fp._deepseek_endpoint() == ("https://gw.example/v1/chat/completions", "deepseek-v3")


def test_kagi_search_auth_header_and_result_filter():
    http = _Http(get_responses=[_Resp(payload={"data": [
        {"t": 0, "url": "https://a", "title": "T", "snippet": "S"},
        {"t": 1, "list": ["相关搜索"]},
    ]})])
    results = fp.kagi_search(http, "KEY", "q")
    assert results == [{"url": "https://a", "title": "T", "snippet": "S"}]
    assert http.get_calls[0]["headers"]["Authorization"] == "Bot KEY"


def test_kagi_search_fail_loud_on_http_error():
    http = _Http(get_responses=[_Resp(status_code=401)])
    with pytest.raises(fp.FactcheckProbeError, match="401"):
        fp.kagi_search(http, "KEY", "q")


def test_extract_facts_rejects_empty_materials():
    with pytest.raises(fp.FactcheckProbeError, match="拒绝无据抽取"):
        fp.extract_facts(_Http(), "d", "某公司", "000001.SZ", [{"url": "u", "text": ""}])


def test_extract_facts_sends_materials_and_parses():
    content = {"q1_net_profit_yi": 1.03, "q1_yoy_pct": 7.95,
               "q1_sources": ["https://a", "https://b"], "risks": [],
               "not_found": [], "contradictions": []}
    http = _Http(post_responses=[_deepseek_resp(content)])
    out = fp.extract_facts(http, "d", "富春环保", "002479.SZ",
                           [{"url": "https://a", "text": "正文A"}])
    assert out["q1_net_profit_yi"] == 1.03
    body = http.post_calls[0]["json"]
    assert body["model"] == fp.DEEPSEEK_MODEL
    assert body["temperature"] == 0
    assert "https://a" in body["messages"][1]["content"]
    assert "禁止使用你自己的记忆" in body["messages"][0]["content"]


def test_extract_facts_fail_loud_on_missing_fields():
    http = _Http(post_responses=[_deepseek_resp({"q1_net_profit_yi": 1.0})])
    with pytest.raises(fp.FactcheckProbeError, match="缺字段"):
        fp.extract_facts(http, "d", "X", "000001.SZ", [{"url": "u", "text": "t"}])


def test_cross_source_confirmed_needs_two_distinct_sources():
    base = {"q1_net_profit_yi": 1.0, "contradictions": []}
    assert fp.cross_source_confirmed({**base, "q1_sources": ["a", "b"]})
    assert not fp.cross_source_confirmed({**base, "q1_sources": ["a", "a"]})
    assert not fp.cross_source_confirmed({**base, "q1_sources": ["a", "b"],
                                          "contradictions": ["数字打架"]})
    assert not fp.cross_source_confirmed({"q1_net_profit_yi": None, "q1_sources": ["a", "b"],
                                          "contradictions": []})


def test_select_candidates_green_only_small_first():
    snapshot = {"decisions": [
        {"ts_code": "1.SZ", "name": "非绿", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED", "TIER_NOT_GREEN"], "evidence": {}},
        {"ts_code": "2.SZ", "name": "大桶", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED"],
         "evidence": {"size_bucket": "大", "score": 0.9}},
        {"ts_code": "3.SZ", "name": "小低分", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED"],
         "evidence": {"size_bucket": "小", "score": 0.5}},
        {"ts_code": "4.SZ", "name": "小高分", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED"],
         "evidence": {"size_bucket": "小", "score": 0.8}},
        {"ts_code": "5.SZ", "name": "持仓", "state": "HOLD",
         "reason_codes": ["HELD"], "evidence": {}},
        {"ts_code": "6.SZ", "name": "污染", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED", "POLLUTION_PENDING_FACTCHECK"],
         "evidence": {"size_bucket": "小", "score": 0.99}},
    ]}
    got = fp.select_candidates(snapshot, limit=2)
    assert [c["ts_code"] for c in got] == ["4.SZ", "3.SZ"]


def test_probe_one_uses_snippet_fallback_and_shapes_record():
    kagi_q1 = _Resp(payload={"data": [{"t": 0, "url": "https://a", "title": "季报", "snippet": "净利1.03亿"}]})
    kagi_risk = _Resp(payload={"data": [{"t": 0, "url": "https://b", "title": "减持", "snippet": "股东减持"}]})
    content = {"q1_net_profit_yi": 1.03, "q1_yoy_pct": None,
               "q1_sources": ["https://a", "https://b"],
               "risks": [{"claim": "股东减持 2%", "source_url": "https://b"}],
               "not_found": ["业绩预告"], "contradictions": []}
    http = _Http(get_responses=[kagi_q1, kagi_risk], post_responses=[_deepseek_resp(content)])
    record = fp.probe_one(http, "k", "d", "富春环保", "002479.SZ",
                          fetcher=lambda _h, _u: "")   # 页面全抓不到 → snippet 兜底
    assert record["confirmed"] is True
    assert record["machine_generated"] is True
    assert record["news"] == ["股东减持 2%(来源:https://b)"]
    assert record["sources"] == ["https://a", "https://b"]


def _tiny_pdf(text: str) -> bytes:
    """用 PyMuPDF 造一页含文字的 PDF,验证 pypdf 抽取路径(两库都在 venv 里)。"""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


def test_pdf_text_extraction():
    data = _tiny_pdf("Net profit 1.03")
    assert "Net profit 1.03" in fp._pdf_text(data)


def test_fetch_page_text_routes_pdf_by_content_type_and_suffix():
    pdf = _tiny_pdf("hello pdf")
    by_type = _Http(get_responses=[_Resp(content=pdf, headers={"content-type": "application/pdf"})])
    assert "hello pdf" in fp.fetch_page_text(by_type, "https://x/notice")
    by_suffix = _Http(get_responses=[_Resp(content=pdf, headers={})])
    assert "hello pdf" in fp.fetch_page_text(by_suffix, "https://x/f.PDF?d=1")


def test_cninfo_announcements_filters_and_builds_urls():
    top = _Resp(payload=[{"code": "002479", "orgId": "org123", "zwjc": "富春环保"}])
    query = _Resp(payload={"announcements": [
        {"announcementTitle": "2026年半年度报告摘要", "adjunctUrl": "finalpage/a.PDF"},
        {"announcementTitle": "关于召开股东大会的通知", "adjunctUrl": "finalpage/b.PDF"},
        {"announcementTitle": "关于控股股东部分股份质押的公告", "adjunctUrl": "finalpage/c.PDF"},
    ]})
    http = _Http(post_responses=[top, query])
    got = fp.cninfo_announcements(http, "002479.SZ", "2026-04-01~2026-08-14")
    assert [x["title"] for x in got] == ["2026年半年度报告摘要", "关于控股股东部分股份质押的公告"]
    assert got[0]["url"] == fp.CNINFO_STATIC + "finalpage/a.PDF"
    body = http.post_calls[1]["data"]
    assert body["stock"] == "002479,org123"
    assert body["column"] == "szse" and body["plate"] == "sz"
    assert body["seDate"] == "2026-04-01~2026-08-14"


def test_cninfo_org_id_fail_loud_when_absent():
    http = _Http(post_responses=[_Resp(payload=[{"code": "999999", "orgId": "x"}])])
    with pytest.raises(fp.FactcheckProbeError, match="orgId"):
        fp.cninfo_org_id(http, "002479.SZ")


def test_main_never_touches_overrides(tmp_path, monkeypatch):
    """端到端(假 HTTP):产出 probe 报告,且绝不写 factcheck_overrides.json。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.local").write_text(
        "KAGI_API_KEY=k\nDEEPSEEK_API_KEY=d\n", encoding="utf-8")
    (tmp_path / "data" / "decisions").mkdir(parents=True)
    (tmp_path / "data" / "factcheck").mkdir(parents=True)
    overrides = tmp_path / "data" / "factcheck_overrides.json"
    overrides.write_text('{"overrides": []}', encoding="utf-8")
    snapshot = {"as_of": "20260814", "decisions": [
        {"ts_code": "002479.SZ", "name": "富春环保", "state": "WAIT",
         "reason_codes": ["FACTCHECK_REQUIRED"],
         "evidence": {"size_bucket": "小", "score": 0.9}}]}
    (tmp_path / "data" / "decisions" / "20260814_buy_decisions.json").write_text(
        json.dumps(snapshot), encoding="utf-8")

    content = {"q1_net_profit_yi": 1.03, "q1_yoy_pct": 7.95,
               "q1_sources": ["https://a", "https://b"], "risks": [],
               "not_found": [], "contradictions": []}
    http = _Http(
        get_responses=[
            _Resp(payload={"data": [{"t": 0, "url": "https://a", "title": "t", "snippet": "s"}]}),
            _Resp(status_code=200, payload={}, text="<html>正文</html>"),
            _Resp(payload={"data": [{"t": 0, "url": "https://b", "title": "t", "snippet": "s"}]}),
            _Resp(status_code=200, payload={}, text="<html>正文2</html>"),
        ],
        post_responses=[_deepseek_resp(content)],
    )
    import sys
    monkeypatch.setitem(sys.modules, "requests", http)

    fp.main(["--no-cninfo"])

    out = json.loads((tmp_path / "data" / "factcheck" / "20260814_probe.json")
                     .read_text(encoding="utf-8"))
    assert out["machine_generated"] is True
    assert out["stocks"]["002479.SZ"]["q1_net_profit_yi"] == 1.03
    # 覆盖文件一个字节都不许动
    assert overrides.read_text(encoding="utf-8") == '{"overrides": []}'
