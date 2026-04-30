import type { ChatResponse } from "../lib/types";
import { SourceList } from "./SourceList";

export type ChatMessage =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string; response: ChatResponse };

export function MessageList({ messages }: { messages: ChatMessage[] }) {
  return (
    <div className="message-list">
      {messages.map((message, index) => (
        <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
          <p>{message.content}</p>
          {message.role === "assistant" ? <SourceList sources={message.response.sources} /> : null}
        </div>
      ))}
    </div>
  );
}
