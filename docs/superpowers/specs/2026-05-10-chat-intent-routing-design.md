# Chat Intent Routing Design

## Goal

The chat experience should feel like a real Seoul Youth Allowance assistant, not a raw RAG endpoint. General conversational inputs such as "안녕?" or "넌 뭐하는 애니?" should receive a concise assistant introduction or scope guidance. Youth Allowance policy questions should continue through the existing query rewrite, multi-query ensemble retrieval, evidence grading, policy resolution, and answer generation graph.

## Scope

This change adds a two-way intent route at the front of the backend graph and updates the API/frontend contract so the UI can distinguish general assistant replies, PDF-backed answers, insufficient evidence, and policy-blocked answers.

Out of scope:

- Long-term chat memory.
- Open-domain general conversation.
- External official web search.
- Reworking the visual layout beyond status-aware rendering.

## Intent Model

The graph uses two top-level intents:

- `general_answer`: greetings, assistant identity questions, capability questions, and off-topic requests.
- `rag`: Youth Allowance policy, eligibility, usage, card, evidence, restriction, payment method, or booklet-related questions.

The classifier should be conservative. If a question might be about Youth Allowance, route it to `rag`. Only clear greetings, identity/capability questions, and clear off-topic requests should route to `general_answer`.

## Graph Flow

```text
START
  -> classify_intent
      -> general_answer -> END
      -> plan_queries
          -> retrieve_pdf
          -> grade_evidence
          -> resolve_policy
              -> generate_blocked_answer
              -> generate_answer
              -> fallback_no_answer
```

`general_answer` does not run retrieval or LLM-based RAG generation. It returns a fixed, product-scoped answer that introduces the assistant or redirects off-topic questions back to Youth Allowance topics.

## API Contract

`ChatResponse.status` should expand from two values to four values:

- `general_answer`: non-PDF conversational guidance from the assistant.
- `answered_from_pdf`: answer generated from sufficient PDF evidence.
- `insufficient_pdf_evidence`: PDF evidence was missing or insufficient.
- `blocked_by_policy`: the question asks about a payment method, transaction type, or purpose that conflicts with a PDF-backed restriction.

Add a separate `intent` field:

- `general_answer`
- `rag`

Response behavior:

```text
general_answer
  sources: []
  needs_external_search: false

answered_from_pdf
  sources: PDF source list
  needs_external_search: false

blocked_by_policy
  sources: PDF source list
  needs_external_search: false

insufficient_pdf_evidence
  sources: []
  needs_external_search: true
```

This keeps policy-blocked answers distinct from normal PDF answers while avoiding the misleading fallback message for greetings and assistant-identity questions.

## General Answer Behavior

Examples:

```text
Q: 안녕?
A: 안녕하세요. 저는 서울 청년수당 참여자 안내책자를 바탕으로 청년수당 사용처, 제한 항목, 카드 결제, 증빙 관련 질문을 도와드리는 챗봇입니다. 궁금한 내용을 물어봐 주세요.
```

```text
Q: 넌 뭐하는 애니?
A: 저는 서울 청년수당 안내를 돕는 챗봇입니다. 안내책자 근거를 바탕으로 사용 가능 항목, 사용 제한, 카드 결제 방식, 증빙 관련 내용을 안내합니다.
```

```text
Q: 오늘 날씨 어때?
A: 저는 청년수당 안내를 돕는 챗봇이라 날씨 같은 일반 정보는 답변하지 않습니다. 청년수당 사용처, 제한 항목, 카드 결제, 증빙 관련 질문을 물어봐 주세요.
```

The answer should never pretend to provide official advice beyond the booklet-backed scope.

## Frontend Changes

Update `ChatResponse.status` and add `intent` in the frontend type definition.

Rendering rules:

- `general_answer`: show only the assistant message; do not show source UI or evidence warning.
- `answered_from_pdf`: show answer and sources.
- `blocked_by_policy`: show answer and sources, with a restriction-oriented label if the current UI has a status label area.
- `insufficient_pdf_evidence`: show answer without sources and keep the "official 안내 확인 필요" style guidance.

No broad UI redesign is required. The change is primarily type-safe status handling and clearer conditional rendering.

## Backend Components

Add or update focused modules:

- `backend/app/graph/state.py`: add `intent`.
- `backend/app/graph/workflow.py`: add `classify_intent`, `general_answer`, and the new conditional edge before `plan_queries`.
- `backend/app/api/schemas.py`: expand `ChatResponse.status` and add `intent`.
- `backend/app/rag/generation.py` or a new small helper: provide fixed general-answer builders.
- `backend/tests/`: add tests for greeting, assistant identity, off-topic, normal RAG routing, and policy-blocked status.

## Testing

Backend tests:

- Greeting input returns `intent="general_answer"`, `status="general_answer"`, no sources, and no external search.
- Assistant identity input returns scoped assistant introduction.
- Clear off-topic input returns a scope redirection, not PDF evidence fallback.
- A Youth Allowance question still reaches retrieval/RAG.
- Policy blocker returns `status="blocked_by_policy"` with PDF sources.
- Existing `answered_from_pdf` and `insufficient_pdf_evidence` cases still pass.

Frontend tests:

- Type contract accepts all four status values.
- General answer renders without sources.
- Blocked policy answer renders sources and a restriction status.
- Insufficient evidence still renders without sources.

## Risks

False positives in intent classification are the main risk. The classifier should prefer `rag` for ambiguous inputs so policy questions are not accidentally answered as general chatter.

The API contract change requires backend and frontend updates in the same implementation slice. Tests should cover both sides before browser testing.
