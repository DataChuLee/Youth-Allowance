import { afterEach, expect, test, vi } from "vitest";

import { sendChatMessage } from "../lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
});

test("sendChatMessage posts question to backend", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      answer: "answer",
      sources: [],
      status: "insufficient_pdf_evidence",
      needs_external_search: true,
    }),
  });
  vi.stubGlobal("fetch", fetchMock);

  const result = await sendChatMessage("question");

  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: "question" }),
  });
  expect(result.answer).toBe("answer");
});
