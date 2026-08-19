"""factcheck 证据抓取管线:Kagi 检索 → 抓原文 → DeepSeek 仅基于抓取材料抽取核对。

边界(spec §5 第四关,candidates.override_status):
- 本脚本只产出**带出处的证据报告**(data/factcheck/<as_of>_probe.json),供人工复核;
  verdict 仍由人工写入 data/factcheck_overrides.json——机器不碰覆盖文件,
  fact-check 只有否决权,报告不构成荐股。
- DeepSeek 只允许基于本脚本抓取并随请求提供的网页材料回答,禁止使用模型记忆——
  事实层里无出处的断言比缺失更危险(与 CLAUDE.md as-of 纪律同源)。
- 密钥(KAGI_API_KEY / DEEPSEEK_API_KEY)由 .env.local 在运行时加载,不入代码不入日志。
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ashare_gauntlet.candidates import HARD_VETO_CODES
from ashare_gauntlet.data.env import load_env_local

DECISION_DIR = "data/decisions"
FACTCHECK_DIR = "data/factcheck"
KAGI_SEARCH_URL = "https://kagi.com/api/v0/search"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"   # 官方端点(默认)
DEEPSEEK_MODEL = "deepseek-chat"
MAX_RESULTS_PER_QUERY = 5
MAX_FETCH_PAGES = 4          # 每股最多抓取的原文页数(控成本)
MAX_PAGE_CHARS = 6000        # 单页截断(控 token)
HTTP_TIMEOUT = 30
# —— 巨潮资讯(法定信息披露平台;公告原文=一手证据,媒体转载只是旁证)——
CNINFO_TOPSEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"
PDF_MAX_CHARS = 10000        # 公告 PDF 截断(定期报告摘要的主要财务数据在前部)
MAX_ANNOUNCEMENTS = 6        # 每股最多入料的公告数(控成本)
MAX_ANNOUNCEMENT_PAGES = 5   # 公告分页上限(150 天×30 条/页,5 页覆盖极端公告密度)
CNINFO_LOOKBACK_DAYS = 150   # 公告回看窗(覆盖一季报以来)
# 入料公告标题关键词:定期报告 + 风险事项(减持/质押/监管);其余公告噪声不入料
ANNOUNCEMENT_KEYWORDS = ("季度报告", "半年度报告", "业绩预告", "业绩快报", "减持",
                         "质押", "问询", "立案", "处罚", "监管函", "关注函")
# 桶排序与 buy_list.SIZE_BUCKET_RANK 同语义(X-08:小桶优先);无桶=中性排最后
_BUCKET_ORDER = {"小": 0, "中": 1, "大": 2}

_EXTRACT_SYSTEM = (
    "你是 A 股事实核查助手。你只能使用 user 消息里提供的网页材料回答,"
    "禁止使用你自己的记忆或训练知识补充任何事实。每一条结论必须附材料中的来源 URL;"
    "材料中找不到的信息必须列入 not_found,不得编造。数字必须与材料原文一致。"
    "输出严格 JSON 对象,字段:q1_net_profit_yi(2026Q1 归母净利润,单位亿元,数值或 null)、"
    "q1_yoy_pct(同比百分数,数值或 null)、q1_sources(支持该数字的 URL 列表)、"
    "risks(列表,每项 {claim, source_url},只收减持/质押/立案/问询/处罚/业绩反转/现金流恶化等风险事实)、"
    "not_found(字符串列表)、contradictions(材料间互相矛盾之处,字符串列表)。"
)


class FactcheckProbeError(RuntimeError):
    """检索/抽取契约被破坏,不能安全产出证据报告。"""


def require_keys(env_path: str = ".env.local") -> tuple[str, str]:
    """加载 .env.local:DeepSeek 密钥必需,Kagi 可选(--sources 模式不需要检索)。"""
    load_env_local(env_path)
    deepseek = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not deepseek:
        raise FactcheckProbeError(".env.local 缺密钥 ['DEEPSEEK_API_KEY']——不产出证据报告")
    return os.environ.get("KAGI_API_KEY", "").strip(), deepseek


def build_queries(name: str, ts_code: str) -> list[str]:
    code6 = ts_code.split(".")[0]
    return [
        f"{name} {code6} 2026年一季报 净利润",
        f"{name} {code6} 减持 质押 立案 问询 处罚",
    ]


def kagi_search(http: Any, api_key: str, query: str) -> list[dict[str, str]]:
    """Kagi Search API;返回 [{url, title, snippet}],非 200 fail-loud。"""
    resp = http.get(KAGI_SEARCH_URL, params={"q": query, "limit": MAX_RESULTS_PER_QUERY},
                    headers={"Authorization": f"Bot {api_key}"}, timeout=HTTP_TIMEOUT)
    if resp.status_code != 200:
        raise FactcheckProbeError(f"Kagi 检索失败 HTTP {resp.status_code}: {query!r}")
    payload = resp.json()
    items = payload.get("data")
    if not isinstance(items, list):
        raise FactcheckProbeError("Kagi 响应缺 data 列表")
    results = []
    for item in items:
        # t==0 为搜索结果条目(t==1 是相关搜索等噪声)
        if isinstance(item, dict) and item.get("t") == 0 and item.get("url"):
            results.append({"url": str(item["url"]), "title": str(item.get("title", "")),
                            "snippet": str(item.get("snippet", ""))})
    return results


def cninfo_org_id(http: Any, ts_code: str) -> str:
    """巨潮 orgId 查询;查不到 fail-loud(法定平台上不存在=输入代码有问题)。"""
    code6 = ts_code.split(".")[0]
    resp = http.post(CNINFO_TOPSEARCH_URL, data={"keyWord": code6, "maxNum": 10},
                     headers={"User-Agent": "Mozilla/5.0 (factcheck-probe)"},
                     timeout=HTTP_TIMEOUT)
    if resp.status_code != 200:
        raise FactcheckProbeError(f"cninfo topSearch 失败 HTTP {resp.status_code}: {ts_code}")
    for item in resp.json() or []:
        if isinstance(item, dict) and item.get("code") == code6 and item.get("orgId"):
            return str(item["orgId"])
    raise FactcheckProbeError(f"cninfo 查不到 {ts_code} 的 orgId")


def cninfo_announcements(http: Any, ts_code: str, se_date: str) -> list[dict[str, str]]:
    """巨潮公告列表(标题关键词过滤后取前 MAX_ANNOUNCEMENTS 条,新在前)。

    公告缺席只能说明关键词窗口内无匹配,不得解读为无风险;API 失败 fail-loud——
    一手证据腿断了必须显式失败,不许静默退回纯媒体口径。
    """
    code6, suffix = ts_code.split(".")
    column, plate = (("szse", "sz") if suffix == "SZ" else ("sse", "sh"))
    org_id = cninfo_org_id(http, ts_code)
    picked: list[dict[str, str]] = []
    # 逐页取(codex P1:150 天窗口公告可超一页,单页截断会漏减持/质押/问询公告)
    for page_num in range(1, MAX_ANNOUNCEMENT_PAGES + 1):
        resp = http.post(CNINFO_QUERY_URL, data={
            "stock": f"{code6},{org_id}", "tabName": "fulltext", "column": column,
            "plate": plate, "pageNum": page_num, "pageSize": 30, "category": "",
            "seDate": se_date, "searchkey": "", "secid": "", "sortName": "",
            "sortType": "", "isHLtitle": "false",
        }, headers={"User-Agent": "Mozilla/5.0 (factcheck-probe)"}, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            raise FactcheckProbeError(f"cninfo 公告查询失败 HTTP {resp.status_code}: {ts_code}")
        payload = resp.json()
        announcements = payload.get("announcements") or []
        for ann in announcements:
            if not isinstance(ann, dict):
                continue
            title = str(ann.get("announcementTitle") or "")
            adjunct = str(ann.get("adjunctUrl") or "")
            if not adjunct or not any(k in title for k in ANNOUNCEMENT_KEYWORDS):
                continue
            picked.append({"title": title, "url": CNINFO_STATIC + adjunct})
            if len(picked) >= MAX_ANNOUNCEMENTS:
                return picked
        if not payload.get("hasMore") and len(announcements) < 30:
            break
    return picked


def _pdf_text(data: bytes) -> str:
    """公告 PDF 文本抽取(pypdf,延迟导入);抽不出文字返回空串由调用方判失败。"""
    import io

    from pypdf import PdfReader

    parts: list[str] = []
    total = 0
    for page in PdfReader(io.BytesIO(data)).pages:
        text = page.extract_text() or ""
        parts.append(text)
        total += len(text)
        if total >= PDF_MAX_CHARS:
            break
    return re.sub(r"\s+", " ", " ".join(parts)).strip()[:PDF_MAX_CHARS]


def fetch_page_text(http: Any, url: str) -> str:
    """抓页面并粗剥标签;失败返回空串(单页失败不整场中断,由材料充分性另行判断)。"""
    try:
        resp = http.get(url, timeout=HTTP_TIMEOUT,
                        headers={"User-Agent": "Mozilla/5.0 (factcheck-probe)"})
        if resp.status_code != 200:
            return ""
        content_type = str(getattr(resp, "headers", {}).get("content-type", "")).lower()
        if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
            return _pdf_text(resp.content)
        text = resp.text
    except Exception:
        return ""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:MAX_PAGE_CHARS]


def _parse_json_content(content: Any) -> dict[str, Any]:
    """容忍网关不支持 response_format 时的 markdown 围栏/前后缀;失败带片段 fail-loud。"""
    if not isinstance(content, str) or not content.strip():
        raise FactcheckProbeError(f"DeepSeek content 非字符串或为空: {content!r}")
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    elif "{" in text:
        text = text[text.index("{"): text.rindex("}") + 1] if "}" in text else text
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise FactcheckProbeError(f"DeepSeek 输出不是 JSON: {exc};开头片段: {content[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise FactcheckProbeError("DeepSeek 输出顶层必须是 JSON 对象")
    return parsed


def _deepseek_endpoint() -> tuple[str, str]:
    """chat 端点与模型名;DEEPSEEK_BASE_URL / DEEPSEEK_MODEL 可走 OpenAI 兼容网关。"""
    base = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
    url = f"{base.rstrip('/')}/chat/completions" if base else DEEPSEEK_CHAT_URL
    model = os.environ.get("DEEPSEEK_MODEL", "").strip() or DEEPSEEK_MODEL
    return url, model


def _deepseek_chat(http: Any, api_key: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    url, model = _deepseek_endpoint()
    resp = http.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0,
              "response_format": {"type": "json_object"}},
        timeout=120,
    )
    if resp.status_code != 200:
        raise FactcheckProbeError(f"DeepSeek 调用失败 HTTP {resp.status_code}(端点 {url})")
    try:
        body = resp.json()
    except ValueError as exc:
        raise FactcheckProbeError(
            f"DeepSeek 响应体不是 JSON(端点 {url});开头片段: {resp.text[:200]!r}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise FactcheckProbeError(f"DeepSeek 响应结构非法: {exc};顶层键: {list(body)[:8]}") from exc
    parsed = _parse_json_content(content)
    if not isinstance(parsed, dict):
        raise FactcheckProbeError("DeepSeek 输出顶层必须是 JSON 对象")
    return parsed


def extract_facts(http: Any, api_key: str, name: str, ts_code: str,
                  materials: list[dict[str, str]]) -> dict[str, Any]:
    """对已抓取材料做抽取;材料为空 fail-loud(空材料抽取=纯模型记忆,禁止)。"""
    usable = [m for m in materials if m.get("text")]
    if not usable:
        raise FactcheckProbeError(f"{ts_code} 无任何可用抓取材料——拒绝无据抽取")
    blocks = "\n\n".join(f"【材料 {i+1}|{m['url']}】\n{m['text']}" for i, m in enumerate(usable))
    user = (f"目标公司:{name}({ts_code})。请基于以下材料完成核查:\n\n{blocks}")
    parsed = _deepseek_chat(http, api_key, [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": user},
    ])
    required = {"q1_net_profit_yi", "q1_sources", "risks", "not_found", "contradictions"}
    missing = sorted(required - set(parsed))
    if missing:
        raise FactcheckProbeError(f"{ts_code} 抽取结果缺字段 {missing}")
    # 类型硬校验(codex P1):字符串会被当可迭代物伪造"双源",畸形输出必须 fail-loud
    for field in ("q1_sources", "risks", "not_found", "contradictions"):
        if not isinstance(parsed[field], list):
            raise FactcheckProbeError(f"{ts_code} 字段 {field} 必须是列表,得到 {type(parsed[field]).__name__}")
    if not all(isinstance(u, str) and u.strip() for u in parsed["q1_sources"]):
        raise FactcheckProbeError(f"{ts_code} q1_sources 含非法来源(须为非空 URL 字符串)")
    for item in parsed["risks"]:
        if not (isinstance(item, dict) and item.get("claim") and item.get("source_url")):
            raise FactcheckProbeError(f"{ts_code} risks 项必须是含 claim/source_url 的对象: {item!r}")
    value = parsed["q1_net_profit_yi"]
    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise FactcheckProbeError(f"{ts_code} q1_net_profit_yi 必须是数值或 null: {value!r}")
    return parsed


def cross_source_confirmed(extraction: dict[str, Any]) -> bool:
    """数字级确认口径:净利数字存在且被 ≥2 个不同来源支持、无矛盾记录。"""
    value = extraction.get("q1_net_profit_yi")
    sources = extraction.get("q1_sources") or []
    distinct = {str(u) for u in sources if u}
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and len(distinct) >= 2 and not extraction.get("contradictions"))


def select_candidates(snapshot: dict[str, Any], limit: int) -> list[dict[str, str]]:
    """从决策快照选出"仅差 fact-check"的 WAIT 候选,小桶优先、同桶按 score 降序。"""
    picked = []
    for d in snapshot.get("decisions", []):
        codes = set(d.get("reason_codes", []))
        if d.get("state") != "WAIT" or "FACTCHECK_REQUIRED" not in codes:
            continue
        # 判据单一来源=candidates.HARD_VETO_CODES:含 SPEC_CROWD/SPIKE_LIMIT 等
        # 写 clear 也解不开的码,避免对🎰/⚡票白跑巨潮+DeepSeek(跨层审计)
        if codes & HARD_VETO_CODES:
            continue
        ev = d.get("evidence", {})
        picked.append({"ts_code": str(d["ts_code"]), "name": str(d.get("name", d["ts_code"])),
                       "_order": (_BUCKET_ORDER.get(ev.get("size_bucket"), 3),
                                  -float(ev.get("score") or 0.0))})
    picked.sort(key=lambda x: x["_order"])
    return [{k: v for k, v in p.items() if k != "_order"} for p in picked[:limit]]


def _record_from_extraction(extraction: dict[str, Any]) -> dict[str, Any]:
    risks = extraction.get("risks") or []
    return {
        "confirmed": cross_source_confirmed(extraction),
        "q1_net_profit_yi": extraction.get("q1_net_profit_yi"),
        "disputes": list(extraction.get("contradictions") or []),
        "news": [f"{r.get('claim')}(来源:{r.get('source_url')})"
                 for r in risks if isinstance(r, dict)],
        "sources": sorted({str(u) for u in (extraction.get("q1_sources") or []) if u}),
        "not_found": list(extraction.get("not_found") or []),
        "machine_generated": True,
    }


def probe_from_urls(http: Any, deepseek_key: str, name: str, ts_code: str,
                    urls: list[str],
                    fetcher: "Callable[[Any, str], str] | None" = None) -> dict[str, Any]:
    """--sources 模式:检索由人/主智能体完成,脚本只抓给定 URL 并做材料内抽取。"""
    fetch = fetcher or fetch_page_text
    materials = [{"url": u, "text": fetch(http, u)} for u in urls]
    extraction = extract_facts(http, deepseek_key, name, ts_code, materials)
    return _record_from_extraction(extraction)


def probe_one(http: Any, kagi_key: str, deepseek_key: str,
              name: str, ts_code: str,
              fetcher: "Callable[[Any, str], str] | None" = None) -> dict[str, Any]:
    if not kagi_key:
        raise FactcheckProbeError("无 KAGI_API_KEY——检索模式不可用,请用 --sources 提供 URL")
    fetch = fetcher or fetch_page_text
    seen: set[str] = set()
    materials: list[dict[str, str]] = []
    for query in build_queries(name, ts_code):
        for result in kagi_search(http, kagi_key, query):
            if result["url"] in seen:
                continue
            seen.add(result["url"])
            if len([m for m in materials if m["text"]]) < MAX_FETCH_PAGES:
                text = fetch(http, result["url"])
            else:
                text = ""
            # 抓不到正文的结果仍以 snippet 入料(snippet 亦来自真实网页,有出处)
            materials.append({"url": result["url"],
                              "text": text or f"[仅摘要] {result['title']} {result['snippet']}"})
    extraction = extract_facts(http, deepseek_key, name, ts_code, materials)
    return _record_from_extraction(extraction)


def latest_decisions_path(decision_dir: str = DECISION_DIR) -> str:
    files = sorted(f for f in os.listdir(decision_dir)
                   if re.fullmatch(r"\d{8}_buy_decisions\.json", f))
    if not files:
        raise FactcheckProbeError(f"{decision_dir} 无决策快照——先跑 scripts.buy_list")
    return os.path.join(decision_dir, files[-1])


def main(argv: "list[str] | None" = None) -> None:
    ap = argparse.ArgumentParser(
        description="factcheck 证据抓取管线(Kagi 检索 + DeepSeek 材料内抽取;不写覆盖文件)")
    ap.add_argument("--decisions", default=None, help="决策快照路径(默认取最新)")
    ap.add_argument("--codes", default=None, help="逗号分隔 ts_code,覆盖自动候选选择")
    ap.add_argument("--limit", type=int, default=8, help="最多核查只数(控成本)")
    ap.add_argument("--no-cninfo", action="store_true",
                    help="跳过巨潮公告腿(默认开启;巨潮=一手公告证据)")
    ap.add_argument("--base-url", default=None,
                    help="临时覆盖 DEEPSEEK_BASE_URL(优先于 .env.local;诊断/纠错用)")
    ap.add_argument("--sources", default=None,
                    help="JSON 文件 {ts_code: [url...]}:检索已由人工/主智能体完成,"
                         "只抓这些 URL 做材料内抽取(无 Kagi 依赖)")
    a = ap.parse_args(argv)

    import requests  # 延迟导入:路径/纯函数消费方不承担网络依赖

    kagi_key, deepseek_key = require_keys()
    if a.base_url:
        os.environ["DEEPSEEK_BASE_URL"] = a.base_url   # CLI 覆盖须晚于 .env.local 加载
    snap_path = a.decisions or latest_decisions_path()
    snapshot = json.load(open(snap_path, encoding="utf-8"))
    as_of = str(snapshot.get("as_of") or "")
    if not re.fullmatch(r"\d{8}", as_of):
        raise FactcheckProbeError(f"决策快照 as_of 非法: {as_of!r}")

    sources: dict[str, list[str]] = {}
    if a.sources:
        raw = json.load(open(a.sources, encoding="utf-8"))
        if not isinstance(raw, dict) or not all(
                isinstance(v, list) and all(isinstance(u, str) for u in v)
                for v in raw.values()):
            raise FactcheckProbeError("--sources 必须是 {ts_code: [url...]} 形状")
        sources = raw

    if a.codes:
        wanted = {c.strip() for c in a.codes.split(",") if c.strip()}
    elif sources:
        wanted = set(sources)
    else:
        wanted = set()
    if wanted:
        by_code = {str(d["ts_code"]): str(d.get("name", d["ts_code"]))
                   for d in snapshot.get("decisions", [])}
        unknown = sorted(wanted - set(by_code))
        if unknown:
            raise FactcheckProbeError(f"指定代码不在快照内: {unknown}")
        candidates = [{"ts_code": c, "name": by_code[c]} for c in sorted(wanted)]
    else:
        candidates = select_candidates(snapshot, a.limit)
    if not candidates:
        print("无'仅差 fact-check'的候选——本次无事可做(允许没有 BUY)")
        return

    report: dict[str, Any] = {}
    unverified: list[str] = []
    now = datetime.now(timezone(timedelta(hours=8)))
    start = (datetime.strptime(as_of, "%Y%m%d")
             - timedelta(days=CNINFO_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    se_date = f"{start}~{datetime.strptime(as_of, '%Y%m%d').strftime('%Y-%m-%d')}"
    for cand in candidates:
        if sources or not kagi_key:
            # sources 模式,或无 Kagi 的全自动模式(材料=纯巨潮公告原文)
            urls = list(sources.get(cand["ts_code"]) or []) if sources else []
            if sources and not urls:
                raise FactcheckProbeError(f"--sources 缺 {cand['ts_code']} 的 URL 列表")
            if not a.no_cninfo:
                anns = cninfo_announcements(requests, cand["ts_code"], se_date)
                urls += [x["url"] for x in anns if x["url"] not in urls]
                print(f"  [{cand['ts_code']}] 巨潮公告入料 {len(anns)} 条: "
                      + ("; ".join(x['title'] for x in anns) or "(关键词窗口内无匹配)"))
            if not urls:
                # 纯巨潮模式下无公告=无一手材料;显式记未核查并以退出码 2 上报
                # (codex P1:静默跳过会让"未核查"与"没有候选"不可区分)
                print(f"  [{cand['ts_code']}] 无可用一手材料——未核查(未核查≠通过)")
                unverified.append(cand["ts_code"])
                continue
            record = probe_from_urls(requests, deepseek_key, cand["name"],
                                     cand["ts_code"], urls)
        else:
            record = probe_one(requests, kagi_key, deepseek_key, cand["name"], cand["ts_code"])
        record["verified_at"] = now.strftime("%Y%m%d")
        report[cand["ts_code"]] = record
        mark = "✓双源" if record["confirmed"] else "⚠需人工"
        print(f"{cand['name']} {cand['ts_code']} {mark} "
              f"Q1净利={record['q1_net_profit_yi']}亿 风险{len(record['news'])}条")

    os.makedirs(FACTCHECK_DIR, exist_ok=True)
    out_path = os.path.join(FACTCHECK_DIR, f"{as_of}_probe.json")
    json.dump({"as_of": as_of, "generated_at": now.isoformat(timespec="seconds"),
               "machine_generated": True,
               "note": "机器证据报告,仅供人工复核;verdict 须人工写入 factcheck_overrides.json",
               "unverified": sorted(unverified),
               "stocks": report},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"→ {out_path}(证据报告;不修改 factcheck_overrides.json,verdict 由人工定)")
    if unverified:
        print(f"⚠ {len(unverified)} 只无一手材料未核查: {','.join(sorted(unverified))}")
        raise SystemExit(2)   # 需人工:未核查绝不解释为通过


if __name__ == "__main__":
    main()
