import type { Source } from "../lib/types";

export function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;

  return (
    <div className="source-list">
      {sources.map((source) => (
        <div className="source-item" key={source.chunk_id}>
          <strong>
            {source.title} p.{source.page}
          </strong>
          <p>{source.excerpt}</p>
        </div>
      ))}
    </div>
  );
}
