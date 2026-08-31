import { useEffect, useState } from "react";
import type { ChatResponse } from "../lib/types";
import { AnswerContent } from "./AnswerContent";
import { SourceList } from "./SourceList";

export type ChatMessage =
  | { role: "user"; content: string }
  | {
      role: "assistant";
      content: string;
      response?: ChatResponse;
      isStreaming?: boolean;
    };

export function MessageList({
  isLoading = false,
  messages,
}: {
  isLoading?: boolean;
  messages: ChatMessage[];
}) {
  const loadingMessage = getLoadingMessage(getLatestUserQuestion(messages));
  const hasEmptyStreamingAssistant = messages.some(
    (message) => message.role === "assistant" && message.isStreaming && !message.content,
  );

  if (messages.length === 0) {
    return (
      <div className="message-list">
        <div className="empty-chat">
          <span>안내 대기 중</span>
          <p>청년수당 사용, 제출, 자격 기준을 물어보세요.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((message, index) => {
        const isStreamingPlaceholder =
          message.role === "assistant" && message.isStreaming && !message.content;

        return (
          <div
            aria-live={isStreamingPlaceholder ? "polite" : undefined}
            className={`message ${message.role}${isStreamingPlaceholder ? " loading" : ""}`}
            key={`${message.role}-${index}`}
            role={isStreamingPlaceholder ? "status" : undefined}
          >
            {message.role === "assistant" ? (
              <>
                {message.content ? (
                  <AnswerContent content={message.content} />
                ) : message.isStreaming ? (
                  <LoadingContent message={loadingMessage} />
                ) : null}
                {message.response ? (
                  <SourceList
                    sources={message.response.sources}
                    status={message.response.status}
                  />
                ) : null}
              </>
            ) : (
              <p>{message.content}</p>
            )}
          </div>
        );
      })}
      {isLoading && !hasEmptyStreamingAssistant ? (
        <div aria-live="polite" className="message assistant loading" role="status">
          <LoadingContent message={loadingMessage} />
        </div>
      ) : null}
    </div>
  );
}

const LOADING_STAGES = [
  { label: "질문을 분석하고 있습니다", duration: 2000 },
  { label: "관련 내용을 검색하고 있습니다", duration: 3000 },
];

function LoadingContent({ message }: { message: string }) {
  const [stageIndex, setStageIndex] = useState(0);
  const [completedStages, setCompletedStages] = useState<string[]>([]);

  useEffect(() => {
    if (stageIndex >= LOADING_STAGES.length) return;

    const timer = setTimeout(() => {
      setCompletedStages((prev) => [...prev, LOADING_STAGES[stageIndex].label]);
      setStageIndex((prev) => prev + 1);
    }, LOADING_STAGES[stageIndex].duration);

    return () => clearTimeout(timer);
  }, [stageIndex]);

  const currentMessage =
    stageIndex < LOADING_STAGES.length ? LOADING_STAGES[stageIndex].label : message;

  return (
    <div className="loading-content">
      {completedStages.map((stage, index) => (
        <p key={index} className="loading-stage-done">
          ✓ {stage}
        </p>
      ))}
      <div className="loading-current">
        <span className="loading-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <p>{currentMessage}</p>
      </div>
    </div>
  );
}

function getLatestUserQuestion(messages: ChatMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];

    if (message.role === "user") {
      return message.content;
    }
  }

  return "";
}

function getLoadingMessage(question: string): string {
  const normalized = question.toLowerCase();

  if (includesAny(normalized, ["배달", "배민", "민족", "요기요", "쿠팡이츠", "음식"])) {
    return "안내책자에서 배달앱·식비 결제 기준을 확인 중입니다.";
  }

  if (includesAny(normalized, ["카카오페이", "네이버페이", "간편결제", "페이", "상품권", "기프티콘"])) {
    return "안내책자에서 간편결제와 체크카드 결제 기준을 확인 중입니다.";
  }

  if (includesAny(normalized, ["현금", "계좌이체", "이체", "인출"])) {
    return "안내책자에서 현금 사용과 증빙 기준을 확인 중입니다.";
  }

  if (includesAny(normalized, ["활동기록서", "제출", "보고서", "증빙", "소명"])) {
    return "안내책자에서 제출 서류와 증빙 기준을 확인 중입니다.";
  }

  if (includesAny(normalized, ["자격", "상실", "중단", "취업", "소득"])) {
    return "안내책자에서 참여 자격과 중단 기준을 확인 중입니다.";
  }

  if (includesAny(normalized, ["사용처", "사용", "가능", "업종", "제한", "결제"])) {
    return "안내책자에서 사용 가능 항목과 제한 업종 기준을 확인 중입니다.";
  }

  return "안내책자에서 질문과 관련된 근거를 확인 중입니다.";
}

function includesAny(value: string, terms: string[]): boolean {
  return terms.some((term) => value.includes(term));
}
