import type { ChatResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function sendChatMessage(question: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      message: "서버 응답을 읽을 수 없습니다.",
    }));
    throw new Error(error.message ?? "요청에 실패했습니다.");
  }

  return response.json();
}
