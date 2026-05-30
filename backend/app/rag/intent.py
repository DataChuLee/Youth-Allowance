from collections.abc import Callable

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.graph.state import IntentDecision
from app.rag.prompts import (
    INTENT_CLASSIFIER_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_USER_PROMPT,
)

ClassifyIntent = Callable[[str], IntentDecision | dict]


def normalize_intent_decision(decision: IntentDecision | dict) -> IntentDecision:
    try:
        return IntentDecision.model_validate(decision)
    except (TypeError, ValidationError):
        return IntentDecision(intent="rag", reason="invalid classifier output")


def create_intent_classifier(settings: Settings | None = None) -> ClassifyIntent:
    resolved_settings = settings or get_settings()
    llm = ChatOpenAI(
        api_key=resolved_settings.openai_api_key,
        model=resolved_settings.openai_chat_model,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INTENT_CLASSIFIER_SYSTEM_PROMPT),
            ("human", INTENT_CLASSIFIER_USER_PROMPT),
        ]
    )
    chain = prompt | llm | JsonOutputParser()

    def classify(question: str) -> IntentDecision:
        return normalize_intent_decision(chain.invoke({"question": question}))

    return classify
