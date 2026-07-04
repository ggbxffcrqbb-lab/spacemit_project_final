from __future__ import annotations

from app.rag.knowledge_base import RetrievedChunk


WEAK_OVERLAP_TERMS = {
    "什么", "怎么", "如何", "为什么", "是否", "哪些", "哪里", "那个", "这个",
    "发现", "应该", "下一步", "处理", "情况", "问题", "位置", "附近", "地方",
    "直接", "优先", "建议", "回答", "用户", "需要", "影响", "关系", "现场",
    "这样", "那个", "这个", "先看", "下", "写",
}

DOMAIN_KEYWORDS = (
    "防腐",
    "腐蚀",
    "返锈",
    "锈",
    "涂层",
    "油漆",
    "补漆",
    "附着力",
    "起泡",
    "粉化",
    "裂纹",
    "焊缝",
    "法兰",
    "支架",
    "储罐",
    "管道",
    "阴保",
    "阴极",
    "阳极",
    "杂散电流",
    "保温",
    "cui",
    "动火",
    "受限空间",
    "巡检",
    "检测",
    "复测",
    "海工",
    "海上",
    "风电",
    "涂装",
    "衬里",
    "树脂",
    "腐蚀裕量",
)


def contains_domain_signal(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in DOMAIN_KEYWORDS)


def informative_overlap_terms(terms: list[str]) -> list[str]:
    result: list[str] = []
    for term in terms:
        lowered = term.lower().strip()
        if not lowered or lowered in WEAK_OVERLAP_TERMS:
            continue
        if any("一" <= char <= "鿿" for char in lowered):
            if len(lowered) >= 2:
                result.append(lowered)
        elif len(lowered) >= 3:
            result.append(lowered)
    return result


def query_focus_terms(query: str) -> list[str]:
    lowered = query.lower()
    focus = [keyword for keyword in DOMAIN_KEYWORDS if keyword in lowered]
    extra_terms = ("低点", "底板", "壳体", "顶板", "接管", "边界", "记录", "范围", "趋势")
    for term in extra_terms:
        if term in lowered and term not in focus:
            focus.append(term)
    return focus


def is_strong_hit(query: str, hit: RetrievedChunk) -> bool:
    lowered_query = query.lower().strip()
    title_text = f"{hit.title} {hit.text}".lower()
    if lowered_query and lowered_query in title_text:
        return True

    focus_terms = query_focus_terms(query)
    strong_terms = informative_overlap_terms(hit.overlap_terms)
    matched_focus_terms = [
        term for term in strong_terms
        if any(term in focus or focus in term for focus in focus_terms)
    ]

    if len(matched_focus_terms) >= 2:
        return True
    if matched_focus_terms and hit.score >= 8.0:
        return True
    if focus_terms and hit.score >= 9.0 and any(term in title_text for term in focus_terms):
        return True
    return False


def filter_strong_hits(query: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not contains_domain_signal(query):
        return []
    return [hit for hit in hits if is_strong_hit(query, hit)]
