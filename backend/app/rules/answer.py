"""규칙 결론을 답변 문장과 출처로 옮긴다.

LLM을 쓰지 않는 템플릿이다. 결론 문장이 결정적이어야 핸들러가 정한 결론이 바뀌지 않는다.
"""

from app.api.schemas import Source
from app.rules.handlers import RuleVerdict

VERDICT_SENTENCES = {
    "allowed": "안내책자 기준으로 가능합니다.",
    "conditional": "안내책자 기준으로 조건부로 가능합니다.",
    "prohibited": "안내책자 기준으로 불가합니다.",
}
BOOKLET_TITLE = "청년수당 참여자 안내책자"


def build_rule_answer(verdict: RuleVerdict, domain: str) -> str:
    lines = [VERDICT_SENTENCES[verdict.verdict]]
    if verdict.condition:
        lines.append(f"- 조건: {verdict.condition}")
    if verdict.evidence:
        lines.append(f"- 필요 증빙: {', '.join(verdict.evidence)}")
    # 제외 사유의 예외에 해당해도 다른 신청 요건은 따로 맞아야 한다(handlers.check_eligibility와 같은 기준).
    if domain == "eligibility" and verdict.verdict == "conditional":
        lines.append("- 나이·거주지·소득 등 다른 신청 요건도 함께 충족해야 합니다.")
    lines += [f"- 근거({source['page']}쪽): \"{source['quote']}\"" for source in verdict.sources]
    return "\n".join(lines)


def rule_sources(verdict: RuleVerdict) -> list[Source]:
    # chunk_id의 rule: 접두어로 규칙 경로 답변을 검색 경로 답변과 구분한다(평가에서 규칙 처리율 집계).
    return [
        Source(type="pdf", title=BOOKLET_TITLE, page=int(source["page"]), excerpt=str(source["quote"])[:280],
               chunk_id=f"rule:{verdict.rule_id}", score=1.0)
        for source in verdict.sources
    ]
