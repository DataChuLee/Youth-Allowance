import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import Page from "../app/page";

test("renders chatbot interface", () => {
  render(<Page />);

  expect(screen.getByText("청년수당 안내 챗봇")).toBeInTheDocument();
  expect(screen.getByPlaceholderText("청년수당에 대해 질문하세요")).toBeInTheDocument();
  expect(screen.getByText("청년수당 사용처")).toBeInTheDocument();
});
