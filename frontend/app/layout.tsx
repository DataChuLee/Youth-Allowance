import type { ReactNode } from "react";

export const metadata = {
  title: "Youth Allowance RAG Chatbot",
  description: "PDF-grounded FAQ chatbot for youth allowance participants.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
