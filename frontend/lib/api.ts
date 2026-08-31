import type { ChatResponse, ConversationMessage } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const API_V1 = `${API_BASE_URL}/api/v1`;

type StreamChatHandlers = {
  onToken: (token: string) => void;
};

type ChatRequestOptions = {
  threadId?: string;
  history?: ConversationMessage[];
};

export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function sendChatMessage(
  question: string,
  options: ChatRequestOptions = {},
): Promise<ChatResponse> {
  const response = await fetch(`${API_V1}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildRequestBody(question, options)),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      message: "서버 응답을 읽을 수 없습니다.",
    }));
    throw apiErrorFromPayload(error, "request_failed", "요청에 실패했습니다.");
  }

  return response.json();
}

export async function streamChatMessage(
  question: string,
  handlers: StreamChatHandlers,
  options: ChatRequestOptions = {},
): Promise<ChatResponse> {
  const response = await fetch(`${API_V1}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildRequestBody(question, options)),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      message: "서버 응답을 읽을 수 없습니다.",
    }));
    throw apiErrorFromPayload(error, "request_failed", "요청에 실패했습니다.");
  }

  if (!response.body) {
    throw new Error("스트리밍 응답을 읽을 수 없습니다.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: ChatResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    buffer = consumeStreamFrames(buffer, (event, payload) => {
      if (event === "token") {
        handlers.onToken(String(payload.text ?? ""));
        return;
      }

      if (event === "done") {
        finalResponse = payload as ChatResponse;
        return;
      }

      if (event === "error") {
        throw apiErrorFromPayload(
          payload,
          "generation_error",
          "답변 생성 중 오류가 발생했습니다.",
        );
      }
    });

    if (done) break;
  }

  if (buffer.trim()) {
    consumeStreamFrames(`${buffer}\n\n`, (event, payload) => {
      if (event === "token") {
        handlers.onToken(String(payload.text ?? ""));
      }
      if (event === "done") {
        finalResponse = payload as ChatResponse;
      }
      if (event === "error") {
        throw apiErrorFromPayload(
          payload,
          "generation_error",
          "답변 생성 중 오류가 발생했습니다.",
        );
      }
    });
  }

  if (!finalResponse) {
    throw new Error("스트리밍 응답이 완료되지 않았습니다.");
  }

  return finalResponse;
}

function buildRequestBody(question: string, options: ChatRequestOptions) {
  return {
    question,
    thread_id: options.threadId ?? null,
    history: options.history ?? [],
  };
}

function apiErrorFromPayload(
  payload: Record<string, unknown>,
  fallbackCode: string,
  fallbackMessage: string,
): ApiError {
  return new ApiError(
    String(payload.code ?? fallbackCode),
    String(payload.message ?? fallbackMessage),
  );
}

function consumeStreamFrames(
  buffer: string,
  handleEvent: (event: string, payload: Record<string, unknown>) => void,
): string {
  let remaining = buffer;
  let boundaryIndex = remaining.indexOf("\n\n");

  while (boundaryIndex >= 0) {
    const frame = remaining.slice(0, boundaryIndex);
    remaining = remaining.slice(boundaryIndex + 2);
    handleStreamFrame(frame, handleEvent);
    boundaryIndex = remaining.indexOf("\n\n");
  }

  return remaining;
}

function handleStreamFrame(
  frame: string,
  handleEvent: (event: string, payload: Record<string, unknown>) => void,
) {
  const lines = frame.split(/\r?\n/);
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (dataLines.length === 0) return;
  handleEvent(event, JSON.parse(dataLines.join("\n")));
}
