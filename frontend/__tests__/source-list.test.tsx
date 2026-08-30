import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

import { SourceList } from "../components/SourceList";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

test("renders duplicate chunk sources without React key warnings", () => {
  const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

  render(
    <SourceList
      sources={[
        {
          type: "pdf",
          title: "Policy A",
          page: 12,
          excerpt: "first source",
          chunk_id: "ocr-page-12-block-7",
          score: 0.8,
        },
        {
          type: "pdf",
          title: "Policy B",
          page: 12,
          excerpt: "second source",
          chunk_id: "ocr-page-12-block-7",
          score: 0.8,
        },
      ]}
      status="answered_from_pdf"
    />,
  );

  expect(screen.getByText("Policy A p.12")).toBeInTheDocument();
  expect(screen.getByText("Policy B p.12")).toBeInTheDocument();
  expect(
    consoleError.mock.calls.filter(([message]) =>
      String(message).includes("Encountered two children with the same key"),
    ),
  ).toHaveLength(0);
});
