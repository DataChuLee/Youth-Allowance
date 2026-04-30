"use client";

import { useState } from "react";

import { ChatInput } from "../components/ChatInput";
import { MessageList, type ChatMessage } from "../components/MessageList";
import { QuickQuestionBar } from "../components/QuickQuestionBar";
import { sendChatMessage } from "../lib/api";

export default function Page() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function ask(question: string) {
    setError(null);
    setIsLoading(true);
    setMessages((current) => [...current, { role: "user", content: question }]);
    try {
      const response = await sendChatMessage(question);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: response.answer, response },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "요청에 실패했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="chat-shell">
        <header>
          <h1>청년수당 안내 챗봇</h1>
          <p>주민등록번호, 계좌번호 등 민감정보는 입력하지 마세요.</p>
        </header>
        <QuickQuestionBar onSelect={ask} />
        <MessageList messages={messages} />
        {isLoading ? <p className="status">답변을 생성하는 중입니다.</p> : null}
        {error ? <p className="error">{error}</p> : null}
        <ChatInput disabled={isLoading} onSubmit={ask} />
      </section>
    </main>
  );
}
