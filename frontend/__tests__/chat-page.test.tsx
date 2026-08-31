import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import Page from "../app/page";
import { streamChatMessage } from "../lib/api";

vi.mock("../lib/api", () => ({
  sendChatMessage: vi.fn(),
  streamChatMessage: vi.fn(),
}));

const streamChatMessageMock = vi.mocked(streamChatMessage);

beforeEach(() => {
  streamChatMessageMock.mockReset();
  window.sessionStorage.clear();
});

afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

test("renders chatbot interface", () => {
  render(<Page />);

  expect(screen.getByRole("img", { name: "청년몽땅정보통" })).toBeInTheDocument();
  expect(
    screen.getByRole("heading", { name: "청년수당 궁금한 점을 바로 알려드립니다." }),
  ).toBeInTheDocument();
  expect(screen.queryByText("청년수당 AI")).not.toBeInTheDocument();
  expect(screen.getByText("PDF 안내책자 기반 답변")).toBeInTheDocument();
  expect(screen.getByText("청년수당 궁금한 점을")).toBeInTheDocument();
  expect(screen.getByText("바로 알려드립니다.")).toBeInTheDocument();
  expect(screen.getByLabelText("청년수당 질문")).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/원하는 청년수당 문의 내용을 입력하세요/)).toBeInTheDocument();
  expect(screen.getByText("청년수당 사용처 알려줘")).toBeInTheDocument();
});

test("disables input while a request is in flight", () => {
  streamChatMessageMock.mockReturnValue(new Promise(() => {}));
  render(<Page />);

  fireEvent.click(screen.getByText("청년수당 사용처 알려줘"));

  expect(streamChatMessageMock).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("status")).toHaveTextContent(
    "질문을 분석하고 있습니다",
  );
  expect(screen.queryByText("답변을 생성하는 중입니다.")).not.toBeInTheDocument();
  expect(screen.queryByText("답변을 정리하는 중입니다.")).not.toBeInTheDocument();
  expect(screen.getByLabelText("청년수당 질문")).toBeDisabled();
  expect(screen.getByRole("button", { name: "보내기" })).toBeDisabled();
  expect(screen.queryByText("활동기록서는 어떻게 제출해?")).not.toBeInTheDocument();
});

test("shows topic-specific loading message for delivery payment questions", () => {
  vi.useFakeTimers();
  streamChatMessageMock.mockReturnValue(new Promise(() => {}));
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "청년수당으로 배달의 민족 결제해도 돼?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  act(() => {
    vi.advanceTimersByTime(2000);
  });
  act(() => {
    vi.advanceTimersByTime(3000);
  });

  expect(screen.getByRole("status")).toHaveTextContent(
    "안내책자에서 배달앱·식비 결제 기준을 확인 중입니다.",
  );
  expect(screen.queryByText("답변을 정리하는 중입니다.")).not.toBeInTheDocument();
});

test("renders assistant answer with source after submit", async () => {
  streamChatMessageMock.mockImplementation(async (_question, { onToken }) => {
    onToken("사용처");
    onToken(" 답변");
    return {
    answer: "사용처 답변",
    sources: [
      {
        type: "pdf",
        title: "청년수당 참여자 안내책자",
        page: 12,
        excerpt: "사용처 관련 문단",
        chunk_id: "pdf-page-12-chunk-2",
        score: 0.9,
      },
    ],
    status: "answered_from_pdf",
    needs_external_search: false,
    intent: "rag",
    };
  });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "사용처 알려줘" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  await waitFor(() => {
    expect(screen.getByText("사용처 답변")).toBeInTheDocument();
  });
  expect(screen.getByText("청년수당 참여자 안내책자 p.12")).toBeInTheDocument();
});

test("renders streamed assistant tokens before final sources", async () => {
  let resolveStream: (value: Awaited<ReturnType<typeof streamChatMessage>>) => void = () => {};
  streamChatMessageMock.mockImplementation((_question, { onToken }) => {
    onToken("청년수당은");
    onToken(" 사용 가능합니다.");
    return new Promise((resolve) => {
      resolveStream = resolve;
    });
  });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "사용처 알려줘" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  await waitFor(() => {
    expect(screen.getByText("청년수당은 사용 가능합니다.")).toBeInTheDocument();
  });
  expect(screen.queryByText("청년수당 참여자 안내책자 p.12")).not.toBeInTheDocument();

  resolveStream({
    answer: "청년수당은 사용 가능합니다.",
    sources: [
      {
        type: "pdf",
        title: "청년수당 참여자 안내책자",
        page: 12,
        excerpt: "사용처 관련 문단",
        chunk_id: "pdf-page-12-chunk-2",
        score: 0.9,
      },
    ],
    status: "answered_from_pdf",
    needs_external_search: false,
    intent: "rag",
  });

  await waitFor(() => {
    expect(screen.getByText("청년수당 참여자 안내책자 p.12")).toBeInTheDocument();
  });
});

test("does not append a later answer to a partial failed stream", async () => {
  streamChatMessageMock
    .mockImplementationOnce(async (_question, { onToken }) => {
      onToken("부분 답변");
      throw new Error("스트리밍 오류");
    })
    .mockImplementationOnce(async (_question, { onToken }) => {
      onToken("새 답변");
      return {
        answer: "새 답변",
        sources: [],
        status: "answered_from_pdf",
        needs_external_search: false,
        intent: "rag",
      };
    });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "첫 질문" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  await waitFor(() => {
    expect(screen.getByText("부분 답변")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("스트리밍 오류");
  });

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "두 번째 질문" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  await waitFor(() => {
    expect(screen.getByText("새 답변")).toBeInTheDocument();
  });
  expect(screen.queryByText("부분 답변새 답변")).not.toBeInTheDocument();
});

test("renders assistant markdown markers as readable answer content", async () => {
  streamChatMessageMock.mockImplementation(async (_question, { onToken }) => {
    const answer =
      "청년수당은 참여자의 진로탐색과 구직활동에 필요한 비용으로 사용할 수 있습니다.\n\n1. **주거비**: 월세와 관리비\n2. **생활비**: 식비와 교통비";
    onToken(answer);
    return {
    answer:
      "청년수당은 참여자의 진로탐색과 구직활동에 필요한 비용으로 사용할 수 있습니다.\n\n1. **주거비**: 월세와 관리비\n2. **생활비**: 식비와 교통비",
    sources: [],
    status: "answered_from_pdf",
    needs_external_search: false,
    intent: "rag",
    };
  });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "청년수당 사용처 알려줘" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  await waitFor(() => {
    expect(screen.getByText("주거비")).toBeInTheDocument();
  });

  expect(screen.queryByText(/\*\*주거비\*\*/)).not.toBeInTheDocument();
  expect(screen.getByText("월세와 관리비")).toBeInTheDocument();
  expect(screen.getByText("생활비")).toBeInTheDocument();
  expect(screen.getByText("식비와 교통비")).toBeInTheDocument();
});

test("keeps source excerpts collapsed until evidence is opened", async () => {
  streamChatMessageMock.mockImplementation(async (_question, { onToken }) => {
    onToken("체크카드 직접 결제 조건으로 사용할 수 있습니다.");
    return {
    answer: "체크카드 직접 결제 조건으로 사용할 수 있습니다.",
    sources: [
      {
        type: "pdf",
        title: "청년수당 참여자 안내책자",
        page: 12,
        excerpt: "청년수당 지원금은 반드시 체크카드로 사용해야 합니다.",
        chunk_id: "pdf-page-12-chunk-2",
        score: 0.9,
      },
    ],
    status: "answered_from_pdf",
    needs_external_search: false,
    intent: "rag",
    };
  });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "배달앱 결제 가능해?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  await waitFor(() => {
    expect(screen.getByText("청년수당 참여자 안내책자 p.12")).toBeInTheDocument();
  });

  expect(
    screen.queryByText("청년수당 지원금은 반드시 체크카드로 사용해야 합니다."),
  ).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "근거 보기" }));

  expect(
    screen.getByText("청년수당 지원금은 반드시 체크카드로 사용해야 합니다."),
  ).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "접기" }));

  expect(
    screen.queryByText("청년수당 지원금은 반드시 체크카드로 사용해야 합니다."),
  ).not.toBeInTheDocument();
});

test("renders general answer without source list", async () => {
  streamChatMessageMock.mockImplementation(async (_question, { onToken }) => {
    onToken("안녕하세요. 청년수당 안내를 도와드리는 챗봇입니다.");
    return {
    answer: "안녕하세요. 청년수당 안내를 도와드리는 챗봇입니다.",
    sources: [],
    status: "general_answer",
    needs_external_search: false,
    intent: "general_answer",
    };
  });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "안녕?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  await waitFor(() => {
    expect(screen.getByText("안녕하세요. 청년수당 안내를 도와드리는 챗봇입니다.")).toBeInTheDocument();
  });
  expect(screen.queryByText(/참여자 안내책자 p\./)).not.toBeInTheDocument();
  expect(screen.queryByText("사용 제한 근거")).not.toBeInTheDocument();
});

test("renders blocked policy answer with restriction label and sources", async () => {
  streamChatMessageMock.mockImplementation(async (_question, { onToken }) => {
    onToken("간편결제는 사용하기 어렵습니다.");
    return {
    answer: "간편결제는 사용하기 어렵습니다.",
    sources: [
      {
        type: "pdf",
        title: "청년수당 참여자 안내책자",
        page: 7,
        excerpt: "카카오페이, 네이버페이 등 간편결제 불가",
        chunk_id: "pdf-page-7-chunk-1",
        score: 0.9,
      },
    ],
    status: "blocked_by_policy",
    needs_external_search: false,
    intent: "rag",
    };
  });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "청년수당으로 카카오페이 결제해도 돼?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));

  await waitFor(() => {
    expect(screen.getByText("간편결제는 사용하기 어렵습니다.")).toBeInTheDocument();
  });
  expect(screen.getByText("사용 제한 근거")).toBeInTheDocument();
  expect(screen.getByText("청년수당 참여자 안내책자 p.7")).toBeInTheDocument();
});

test("keeps one thread id and sends bounded history within the same tab", async () => {
  streamChatMessageMock.mockImplementation(async (question, { onToken }) => {
    const answer = `${question} 답변`;
    onToken(answer);
    return {
      answer,
      sources: [],
      status: "answered_from_pdf",
      needs_external_search: false,
      intent: "rag",
    };
  });
  render(<Page />);

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "첫 질문" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));
  await waitFor(() => expect(screen.getByText("첫 질문 답변")).toBeInTheDocument());

  fireEvent.change(screen.getByLabelText("청년수당 질문"), {
    target: { value: "두 번째 질문" },
  });
  fireEvent.click(screen.getByRole("button", { name: "보내기" }));
  await waitFor(() => expect(screen.getByText("두 번째 질문 답변")).toBeInTheDocument());

  const firstOptions = streamChatMessageMock.mock.calls[0][2];
  const secondOptions = streamChatMessageMock.mock.calls[1][2];
  expect(firstOptions?.threadId).toBeTruthy();
  expect(secondOptions?.threadId).toBe(firstOptions?.threadId);
  expect(firstOptions?.history).toEqual([]);
  expect(secondOptions?.history).toEqual([
    { role: "user", content: "첫 질문" },
    { role: "assistant", content: "첫 질문 답변" },
  ]);
});
