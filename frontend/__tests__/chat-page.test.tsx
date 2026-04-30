import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import Page from "../app/page";
import { sendChatMessage } from "../lib/api";

vi.mock("../lib/api", () => ({
  sendChatMessage: vi.fn(),
}));

const sendChatMessageMock = vi.mocked(sendChatMessage);

beforeEach(() => {
  sendChatMessageMock.mockReset();
});

afterEach(() => {
  cleanup();
});

test("renders chatbot interface", () => {
  render(<Page />);

  expect(screen.getByText("청년수당 안내 챗봇")).toBeInTheDocument();
  expect(screen.getByLabelText("청년수당 질문")).toBeInTheDocument();
  expect(screen.getByPlaceholderText("청년수당에 대해 질문하세요")).toBeInTheDocument();
  expect(screen.getByText("청년수당 사용처")).toBeInTheDocument();
});

test("disables quick questions and input while a request is in flight", () => {
  sendChatMessageMock.mockReturnValue(new Promise(() => {}));
  render(<Page />);

  fireEvent.click(screen.getByText("청년수당 사용처"));
  fireEvent.click(screen.getByText("활동기록서 제출"));

  expect(sendChatMessageMock).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("status")).toHaveTextContent("답변을 생성하는 중입니다.");
  expect(screen.getByLabelText("청년수당 질문")).toBeDisabled();
  expect(screen.getByText("활동기록서 제출")).toBeDisabled();
});

test("renders assistant answer with source after submit", async () => {
  sendChatMessageMock.mockResolvedValue({
    answer: "사용처 답변",
    sources: [
      {
        type: "pdf",
        title: "청년수당 참여자 안내책자",
        page: 12,
        excerpt: "사용처 관련 문단",
        chunk_id: "pdf-page-12-chunk-2",
      },
    ],
    status: "answered_from_pdf",
    needs_external_search: false,
  });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "사용처 알려줘" },
  });
  fireEvent.click(screen.getByText("전송"));

  await waitFor(() => {
    expect(screen.getByText("사용처 답변")).toBeInTheDocument();
  });
  expect(screen.getByText("청년수당 참여자 안내책자 p.12")).toBeInTheDocument();
});
