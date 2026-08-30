import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, test } from "vitest";

import { AnswerContent } from "../components/AnswerContent";

afterEach(() => {
  cleanup();
});

test("renders repeated numbered headings as one ordered list with nested evidence bullets", () => {
  const { container } = render(
    <AnswerContent
      content={[
        "1. 월세",
        "- 임대차계약서",
        "- 이체확인증",
        "1. 관리비",
        "- 고지서",
        "1. 공과금",
        "- 고지서",
        "1. 주거 관련 대출",
        "- 원금 및 이자 증빙",
      ].join("\n")}
    />,
  );

  const orderedLists = container.querySelectorAll(".answer-content > ol");
  expect(orderedLists).toHaveLength(1);

  const topLevelItems = orderedLists[0].querySelectorAll(":scope > li");
  expect(topLevelItems).toHaveLength(4);
  expect(topLevelItems[0]).toHaveTextContent("월세");
  expect(topLevelItems[1]).toHaveTextContent("관리비");
  expect(topLevelItems[2]).toHaveTextContent("공과금");
  expect(topLevelItems[3]).toHaveTextContent("주거 관련 대출");

  const firstNestedItems = topLevelItems[0].querySelectorAll("ul > li");
  expect(firstNestedItems).toHaveLength(2);
  expect(firstNestedItems[0]).toHaveTextContent("임대차계약서");
  expect(firstNestedItems[1]).toHaveTextContent("이체확인증");
});

test("keeps markdown bold titles readable inside list items", () => {
  render(
    <AnswerContent
      content={[
        "1. **주거비**: 월세와 관리비",
        "- **월세**: 임대차계약서",
      ].join("\n")}
    />,
  );

  expect(screen.queryByText(/\*\*주거비\*\*/)).not.toBeInTheDocument();
  expect(screen.getByText("주거비")).toBeInTheDocument();
  expect(screen.getByText("월세와 관리비")).toBeInTheDocument();
  expect(screen.getByText("월세")).toBeInTheDocument();
  expect(screen.getByText("임대차계약서")).toBeInTheDocument();
});
