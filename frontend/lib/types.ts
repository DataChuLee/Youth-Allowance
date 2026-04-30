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
  status: "answered_from_pdf" | "insufficient_pdf_evidence";
  needs_external_search: boolean;
};
