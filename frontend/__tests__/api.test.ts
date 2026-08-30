import { afterEach, expect, test, vi } from "vitest";

import { ApiError, sendChatMessage, streamChatMessage } from "../lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
});

test("sendChatMessage posts question to /api/v1/chat", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      answer: "answer",
      sources: [],
      status: "insufficient_pdf_evidence",
      needs_external_search: true,
      intent: "rag",
    }),
  });
  vi.stubGlobal("fetch", fetchMock);

  const result = await sendChatMessage("question");

  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: "question", thread_id: null, history: [] }),
  });
  expect(result.answer).toBe("answer");
  expect(result.intent).toBe("rag");
});

test("sendChatMessage forwards thread_id when provided", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      answer: "answer",
      sources: [],
      status: "answered_from_pdf",
      needs_external_search: false,
      intent: "rag",
    }),
  });
  vi.stubGlobal("fetch", fetchMock);

  await sendChatMessage("question", {
    threadId: "session-abc",
    history: [{ role: "user", content: "previous question" }],
  });

  expect(fetchMock).toHaveBeenCalledWith(
    "http://localhost:8000/api/v1/chat",
    expect.objectContaining({
      body: JSON.stringify({
        question: "question",
        thread_id: "session-abc",
        history: [{ role: "user", content: "previous question" }],
      }),
    }),
  );
});

test("streamChatMessage reads token events and returns final response", async () => {
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      controller.enqueue(
        encoder.encode(
          [
            'event: token\ndata: {"text":"사용처"}',
            'event: token\ndata: {"text":" 답변"}',
            [
              "event: done",
              'data: {"answer":"사용처 답변","sources":[],"status":"answered_from_pdf","needs_external_search":false,"intent":"rag"}',
            ].join("\n"),
          ].join("\n\n"),
        ),
      );
      controller.close();
    },
  });
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    body: stream,
  });
  const tokens: string[] = [];
  vi.stubGlobal("fetch", fetchMock);

  const result = await streamChatMessage("question", {
    onToken: (token) => tokens.push(token),
  });

  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: "question", thread_id: null, history: [] }),
  });
  expect(tokens).toEqual(["사용처", " 답변"]);
  expect(result.answer).toBe("사용처 답변");
  expect(result.status).toBe("answered_from_pdf");
});

test("streamChatMessage uses the common SSE error schema", async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(
        new TextEncoder().encode(
          'event: error\ndata: {"code":"generation_error","message":"생성 실패"}\n\n',
        ),
      );
      controller.close();
    },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, body: stream }));

  const error = await streamChatMessage("question", { onToken: vi.fn() }).catch(
    (caught) => caught,
  );

  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({ code: "generation_error", message: "생성 실패" });
});

test("sendChatMessage uses the common HTTP error schema", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ code: "invalid_request", message: "잘못된 요청" }),
    }),
  );

  const error = await sendChatMessage("question").catch((caught) => caught);

  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({ code: "invalid_request", message: "잘못된 요청" });
});
