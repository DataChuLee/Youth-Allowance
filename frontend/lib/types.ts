export type Source = {
  type: "pdf";
  title: string;
  page: number;
  excerpt: string;
  chunk_id: string;
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
