import { useState } from "react";

import type { ChatResponse, Source } from "../lib/types";

export function SourceList({
  sources,
  status,
}: {
  sources: Source[];
  status: ChatResponse["status"];
}) {
  if (sources.length === 0) return null;

  return (
    <div className="source-list">
      {status === "blocked_by_policy" ? (
        <div className="source-label">사용 제한 근거</div>
      ) : null}
      {sources.map((source) => (
        <SourceItem key={source.chunk_id} source={source} />
      ))}
    </div>
  );
}

function SourceItem({ source }: { source: Source }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="source-item">
      <div className="source-item-header">
        <strong>
          {source.title} p.{source.page}
        </strong>
        <button
          aria-expanded={isOpen}
          className="source-toggle"
          onClick={() => setIsOpen((current) => !current)}
          type="button"
        >
          {isOpen ? "접기" : "근거 보기"}
        </button>
      </div>
      {isOpen ? <p className="source-excerpt">{source.excerpt}</p> : null}
    </div>
  );
}
