from collections.abc import Callable

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.rag.prompts import QUERY_PLANNER_SYSTEM_PROMPT, QUERY_PLANNER_USER_PROMPT


class QueryPlan(BaseModel):
    rewritten_query: str = ""
    queries: list[str] = Field(default_factory=list)


PlanQuery = Callable[[str], QueryPlan | dict]
# "X에 써도 되나요?"는 사용 가능 항목 청크와 가깝게 임베딩되어 사용 불가 업종 목록이 검색되지 않는다.
RESTRICTION_QUERY_PREFIX = "청년수당 클린카드 사용 불가 업종"


def normalize_query(query: object) -> str:
    return " ".join(str(query).split())


def create_query_planner(settings: Settings | None = None) -> PlanQuery:
    resolved_settings = settings or get_settings()
    llm = ChatOpenAI(
        api_key=resolved_settings.openai_api_key,
        model=resolved_settings.openai_chat_model,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUERY_PLANNER_SYSTEM_PROMPT),
            ("human", QUERY_PLANNER_USER_PROMPT),
        ]
    )
    chain = prompt | llm | JsonOutputParser()

    def plan_query(question: str) -> QueryPlan:
        return QueryPlan.model_validate(chain.invoke({"question": question}))

    return plan_query


def build_search_queries(
    question: str,
    plan_query: PlanQuery | None = None,
    settings: Settings | None = None,
    max_queries: int = 6,
) -> list[str]:
    resolved_plan_query = plan_query or create_query_planner(settings)
    planned = QueryPlan.model_validate(resolved_plan_query(question))
    # 사용 불가 목록 대조 질의는 원 질문 바로 뒤에 둔다. 검색 결과를 질의 순서대로 번갈아 합치고
    # 문서 수를 자르므로, 뒤에 두면 잘려 나간다.
    candidates = [
        question,
        f"{RESTRICTION_QUERY_PREFIX} {planned.rewritten_query or question}",
        planned.rewritten_query,
        *planned.queries,
    ]

    unique_queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_query(candidate)
        if not normalized or normalized in seen:
            continue
        unique_queries.append(normalized)
        seen.add(normalized)

    return unique_queries[:max_queries]
