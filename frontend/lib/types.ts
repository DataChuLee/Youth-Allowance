export type Source = {
  type: string;
  title: string;
  page: number;
  excerpt: string;
  chunk_id: string;
  score: number;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  status:
    | "general_answer"
    | "answered_from_pdf"
    | "insufficient_pdf_evidence"
    | "blocked_by_policy";
  needs_external_search: boolean;
  intent: "general_answer" | "rag";
};

export type ConversationMessage = {
  role: "user" | "assistant";
  content: string;
};
