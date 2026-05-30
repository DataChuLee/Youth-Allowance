import type { ReactNode } from "react";

import "./globals.css";

export const metadata = {
  title: "청년몽땅정보통 청년수당 안내",
  description: "서울시 청년수당 참여자 안내책자 기반 채팅 안내 서비스입니다.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
