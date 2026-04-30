import { expect, test, vi } from "vitest";

import { sendChatMessage } from "../lib/api";

test("sendChatMessage posts question to backend", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      answer: "답변",
      sources: [],
      status: "insufficient_pdf_evidence",
      needs_external_search: true,
    }),
  });
  vi.stubGlobal("fetch", fetchMock);

  const result = await sendChatMessage("질문");

  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: "질문" }),
  });
  expect(result.answer).toBe("답변");
});
