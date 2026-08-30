"use client";

import { useRef, useState } from "react";

import { ChatInput } from "../components/ChatInput";
import { MessageList, type ChatMessage } from "../components/MessageList";
import { QuickQuestionBar } from "../components/QuickQuestionBar";
import { streamChatMessage } from "../lib/api";
import { getOrCreateThreadId } from "../lib/session";
import type { ConversationMessage } from "../lib/types";

export default function Page() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inFlightRef = useRef(false);
  const threadIdRef = useRef<string | null>(null);

  async function ask(question: string) {
    if (inFlightRef.current) return;

    inFlightRef.current = true;
    const history = buildConversationHistory(messages);
    const threadId = threadIdRef.current ?? getOrCreateThreadId();
    threadIdRef.current = threadId;
    setError(null);
    setIsLoading(true);
    setMessages((current) => [
      ...current,
      { role: "user", content: question },
      { role: "assistant", content: "", isStreaming: true },
    ]);

    try {
      const response = await streamChatMessage(
        question,
        {
          onToken: (token) => {
            setMessages((current) =>
              updateStreamingAssistant(current, (message) => ({
                ...message,
                content: message.content + token,
              })),
            );
          },
        },
        { threadId, history },
      );
      setMessages((current) =>
        updateStreamingAssistant(current, (message) => ({
          ...message,
          content: response.answer || message.content,
          response,
          isStreaming: false,
        })),
      );
    } catch (err) {
      setMessages(removeEmptyStreamingAssistant);
      setError(err instanceof Error ? err.message : "요청에 실패했습니다.");
    } finally {
      inFlightRef.current = false;
      setIsLoading(false);
    }
  }

  return (
    <div className="app-frame">
      <header className="brand-header" aria-label="서비스 상단">
        <div className="brand-header-inner">
          <div className="logo-plate">
            <img
              alt="청년몽땅정보통"
              className="brand-logo"
              height="64"
              src="/logo.png"
              width="240"
            />
          </div>
          <strong className="assistant-title">청년수당 AI Assistant</strong>
        </div>
      </header>

      <main className="chat-page" aria-labelledby="chat-title">
        {messages.length === 0 ? (
          <section className="intro-card" aria-labelledby="chat-title">
            <p className="intro-kicker">PDF 안내책자 기반 답변</p>
            <h1 id="chat-title">
              청년수당 궁금한 점을
              <span>바로 알려드립니다.</span>
            </h1>
            <p className="intro-copy">
              사용처, 활동기록서, 자격상실처럼 참여자가 자주 묻는 내용을 자연스럽게
              이어 물어보세요. 구체적으로 입력할수록 더 정확하게 답변합니다.
            </p>

            <div className="feature-grid" aria-label="안내 영역">
              <article>
                <span aria-hidden="true">💳</span>
                <strong>사용처 안내</strong>
                <p>카드 사용, 현금 사용, 제한 업종 기준 확인</p>
              </article>
              <article>
                <span aria-hidden="true">📝</span>
                <strong>제출 안내</strong>
                <p>활동기록서와 참여 중 제출해야 할 내용 안내</p>
              </article>
              <article>
                <span aria-hidden="true">🔎</span>
                <strong>자격 확인</strong>
                <p>자격상실, 참여 중단, 유의사항 탐색</p>
              </article>
            </div>

            <QuickQuestionBar disabled={isLoading} onSelect={ask} />
          </section>
        ) : (
          <section className="conversation-panel" aria-label="청년수당 안내 채팅">
            <MessageList isLoading={isLoading} messages={messages} />
          </section>
        )}

        <div className="input-dock">
          <div className="input-dock-inner">
            <ChatInput disabled={isLoading} onSubmit={ask} />
          </div>

          {error ? (
            <p className="error" role="alert">
              {error}
            </p>
          ) : null}
        </div>
      </main>
    </div>
  );
}

function buildConversationHistory(messages: ChatMessage[]): ConversationMessage[] {
  return messages
    .filter(
      (message) =>
        message.content && (message.role === "user" || !message.isStreaming),
    )
    .map((message) => ({ role: message.role, content: message.content }))
    .slice(-4);
}

function updateStreamingAssistant(
  messages: ChatMessage[],
  update: (
    message: Extract<ChatMessage, { role: "assistant" }>,
  ) => Extract<ChatMessage, { role: "assistant" }>,
): ChatMessage[] {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];

    if (message.role === "assistant" && message.isStreaming) {
      const next = [...messages];
      next[index] = update(message);
      return next;
    }
  }

  return messages;
}

function removeEmptyStreamingAssistant(messages: ChatMessage[]): ChatMessage[] {
  return messages.filter(
    (message) => !(message.role === "assistant" && message.isStreaming && !message.content),
  );
}
