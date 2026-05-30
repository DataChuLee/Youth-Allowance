# Chat Intent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLM-based intent routing so general assistant questions get scoped guidance while Youth Allowance questions continue through the RAG graph.

**Architecture:** Add an LCEL intent classifier before query planning. Extend the API contract with `intent` and four statuses, then update frontend types and rendering so `general_answer`, `answered_from_pdf`, `blocked_by_policy`, and `insufficient_pdf_evidence` produce distinct UI behavior.

**Tech Stack:** FastAPI, Pydantic, LangGraph, LangChain LCEL, ChatOpenAI, Next.js, React, TypeScript, pytest, Vitest.

---

### Task 1: Backend Contract and Intent Model

**Files:**
- Modify: `backend/app/api/schemas.py`
- Modify: `backend/app/graph/state.py`
- Test: `backend/tests/test_advanced_rag.py`

- [ ] **Step 1: Write failing tests**

Add tests that expect:

```python
def test_run_chat_graph_returns_general_answer_without_retrieval() -> None:
    def fail_retrieve(_: str) -> list[RetrievedDocument]:
        raise AssertionError("general_answer must not retrieve")

    response = run_chat_graph(
        "안녕?",
        classify_intent=lambda question: IntentDecision(intent="general_answer", reason="greeting"),
        retrieve_documents=fail_retrieve,
    )

    assert response.intent == "general_answer"
    assert response.status == "general_answer"
    assert response.needs_external_search is False
    assert response.sources == []
```

```python
def test_run_chat_graph_returns_blocked_policy_status() -> None:
    response = run_chat_graph(
        "청년수당으로 카카오페이 결제해도 돼?",
        classify_intent=lambda question: IntentDecision(intent="rag", reason="policy question"),
        plan_search_queries=lambda question: ["청년수당 간편결제 불가"],
        retrieve_documents=fake_retrieve,
        grade_evidence=fake_grade,
    )

    assert response.intent == "rag"
    assert response.status == "blocked_by_policy"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
cd backend
..\venv\Scripts\python.exe -m pytest tests\test_advanced_rag.py -v
```

Expected: FAIL because `IntentDecision`, `classify_intent`, `ChatResponse.intent`, `general_answer`, and `blocked_by_policy` are not implemented.

- [ ] **Step 3: Implement contract types**

Update backend models:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: Literal[
        "general_answer",
        "answered_from_pdf",
        "insufficient_pdf_evidence",
        "blocked_by_policy",
    ]
    needs_external_search: bool
    intent: Literal["general_answer", "rag"]
```

```python
class IntentDecision(BaseModel):
    intent: Literal["general_answer", "rag"] = "rag"
    reason: str = ""
```

Add `intent: Literal["general_answer", "rag"] = "rag"` to `GraphState`.

- [ ] **Step 4: Run tests to verify progress**

Run the same pytest command. Expected: remaining failures should be workflow behavior, not missing types.

### Task 2: LCEL Intent Classifier

**Files:**
- Modify: `backend/app/rag/prompts.py`
- Create: `backend/app/rag/intent.py`
- Test: `backend/tests/test_advanced_rag.py`

- [ ] **Step 1: Write failing classifier tests**

Add tests for classifier normalization:

```python
def test_normalize_intent_decision_routes_invalid_intent_to_rag() -> None:
    result = normalize_intent_decision({"intent": "unknown", "reason": "bad"})

    assert result.intent == "rag"
```

```python
def test_run_chat_graph_routes_classifier_failure_to_rag() -> None:
    def fail_classify(_: str) -> IntentDecision:
        raise RuntimeError("classifier failed")

    response = run_chat_graph(
        "청년수당 카드 사용처는?",
        classify_intent=fail_classify,
        retrieve_documents=fake_retrieve,
        generate_answer=fake_generate,
    )

    assert response.intent == "rag"
    assert response.status == "answered_from_pdf"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
cd backend
..\venv\Scripts\python.exe -m pytest tests\test_advanced_rag.py -v
```

Expected: FAIL because `app.rag.intent` does not exist.

- [ ] **Step 3: Implement LCEL classifier**

Create `backend/app/rag/intent.py` with:

```python
def create_intent_classifier(settings: Settings | None = None) -> ClassifyIntent:
    llm = ChatOpenAI(...)
    prompt = ChatPromptTemplate.from_messages([...])
    chain = prompt | llm | JsonOutputParser()
    return lambda question: normalize_intent_decision(chain.invoke({"question": question}))
```

`normalize_intent_decision` must catch invalid intent values and return `IntentDecision(intent="rag", reason="invalid classifier output")`.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same pytest command. Expected: PASS or only workflow integration failures remain.

### Task 3: LangGraph Routing

**Files:**
- Modify: `backend/app/graph/workflow.py`
- Modify: `backend/app/rag/generation.py`
- Test: `backend/tests/test_advanced_rag.py`
- Test: `backend/tests/test_workflow.py`

- [ ] **Step 1: Write failing graph routing tests**

Add tests for:

```python
def test_general_answer_status_is_not_pdf_fallback() -> None:
    response = run_chat_graph(
        "넌 뭐하는 애니?",
        classify_intent=lambda question: IntentDecision(intent="general_answer", reason="identity"),
        retrieve_documents=lambda question: [],
    )

    assert response.status == "general_answer"
    assert "청년수당" in response.answer
    assert response.needs_external_search is False
```

```python
def test_rag_intent_uses_existing_rag_path() -> None:
    response = run_chat_graph(
        "청년수당 카드 사용처는?",
        classify_intent=lambda question: IntentDecision(intent="rag", reason="policy"),
        retrieve_documents=fake_retrieve,
        generate_answer=fake_generate,
    )

    assert response.status == "answered_from_pdf"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
cd backend
..\venv\Scripts\python.exe -m pytest tests\test_advanced_rag.py tests\test_workflow.py -v
```

Expected: FAIL because graph does not branch on intent yet.

- [ ] **Step 3: Implement graph nodes**

Add:

```python
def classify_intent_node(classify_intent: ClassifyIntent) -> Callable[[GraphState], GraphState]:
    ...

def route_intent(state: GraphState) -> Literal["general_answer", "rag"]:
    return state.intent

def general_answer_node() -> Callable[[GraphState], GraphState]:
    ...
```

Insert conditional edge:

```python
START -> classify_intent
classify_intent -> general_answer | plan_queries
```

Update blocked policy branch to set `status="blocked_by_policy"`.

- [ ] **Step 4: Run backend tests**

Run:

```powershell
cd backend
..\venv\Scripts\python.exe -m pytest
```

Expected: PASS.

### Task 4: Frontend Contract and Rendering

**Files:**
- Modify: `frontend/lib/types.ts`
- Modify: `frontend/components/MessageList.tsx`
- Modify: `frontend/components/SourceList.tsx` if needed
- Test: `frontend/__tests__/chat-page.test.tsx`

- [ ] **Step 1: Write failing frontend tests**

Add tests that render:

```ts
status: "general_answer",
intent: "general_answer",
sources: []
```

and expect no source list or evidence warning.

Add tests that render:

```ts
status: "blocked_by_policy",
intent: "rag",
sources: [...]
```

and expect the text `사용 제한 근거`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
cd frontend
npm test
```

Expected: FAIL because frontend types/rendering do not support the new statuses.

- [ ] **Step 3: Update frontend types and UI**

Update `ChatResponse`:

```ts
status:
  | "general_answer"
  | "answered_from_pdf"
  | "insufficient_pdf_evidence"
  | "blocked_by_policy";
intent: "general_answer" | "rag";
```

Render `사용 제한 근거` for `blocked_by_policy`; keep source rendering for `answered_from_pdf` and `blocked_by_policy`; hide sources for `general_answer`.

- [ ] **Step 4: Run frontend tests**

Run:

```powershell
cd frontend
npm test
```

Expected: PASS.

### Task 5: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run backend tests**

```powershell
cd backend
..\venv\Scripts\python.exe -m pytest
```

Expected: PASS.

- [ ] **Step 2: Run frontend tests**

```powershell
cd frontend
npm test
```

Expected: PASS.

- [ ] **Step 3: Run build checks**

```powershell
cd frontend
npm run build
```

Expected: PASS.

- [ ] **Step 4: Manual browser check**

Run backend and frontend, then test:

```text
안녕?
넌 뭐하는 애니?
오늘 날씨 어때?
청년수당으로 배달의민족 결제해도 돼?
청년수당으로 카카오페이 결제해도 돼?
```

Expected:

- Greeting/identity/off-topic return scoped general answers.
- RAG questions return PDF-backed or blocked-policy answers with appropriate source behavior.
