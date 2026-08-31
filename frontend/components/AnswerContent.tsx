import { Fragment, type ReactNode } from "react";

type AnswerBlock =
  | { type: "paragraph"; text: string }
  | { type: "ordered-list" | "unordered-list"; items: AnswerListItem[] };

type AnswerListItem = {
  text: string;
  children: AnswerListItem[];
};

const ORDERED_ITEM_PATTERN = /^\s*\d+[.)]\s+(.+)$/;
const UNORDERED_ITEM_PATTERN = /^\s*[-*•]\s+(.+)$/;
const BOLD_SEGMENT_PATTERN = /(\*\*[^*]+\*\*)/g;

export function AnswerContent({ content }: { content: string }) {
  const blocks = parseAnswerBlocks(content);

  return (
    <div className="answer-content">
      {blocks.map((block, index) => {
        if (block.type === "paragraph") {
          return <p key={`paragraph-${index}`}>{renderInlineText(block.text)}</p>;
        }

        const ListTag = block.type === "ordered-list" ? "ol" : "ul";

        return (
          <ListTag key={`${block.type}-${index}`}>
            {block.items.map((item, itemIndex) => (
              <li key={`${item.text}-${itemIndex}`}>{renderListItem(item)}</li>
            ))}
          </ListTag>
        );
      })}
    </div>
  );
}

function parseAnswerBlocks(content: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  for (const line of lines) {
    const orderedMatch = line.match(ORDERED_ITEM_PATTERN);
    const unorderedMatch = line.match(UNORDERED_ITEM_PATTERN);

    if (orderedMatch) {
      appendListItem(blocks, "ordered-list", orderedMatch[1]);
      continue;
    }

    if (unorderedMatch) {
      appendNestedOrTopLevelBullet(blocks, unorderedMatch[1]);
      continue;
    }

    blocks.push({ type: "paragraph", text: line });
  }

  return blocks.length > 0 ? blocks : [{ type: "paragraph", text: content.trim() }];
}

function appendListItem(
  blocks: AnswerBlock[],
  type: "ordered-list" | "unordered-list",
  item: string,
) {
  const previousBlock = blocks[blocks.length - 1];
  const listItem = createListItem(item);

  if (previousBlock?.type === type) {
    previousBlock.items.push(listItem);
    return;
  }

  blocks.push({ type, items: [listItem] });
}

function appendNestedOrTopLevelBullet(blocks: AnswerBlock[], item: string) {
  const previousBlock = blocks[blocks.length - 1];

  if (previousBlock?.type === "ordered-list" && previousBlock.items.length > 0) {
    previousBlock.items[previousBlock.items.length - 1].children.push(createListItem(item));
    return;
  }

  appendListItem(blocks, "unordered-list", item);
}

function createListItem(text: string): AnswerListItem {
  return { text, children: [] };
}

function renderListItem(item: AnswerListItem): ReactNode {
  return (
    <>
      {renderListItemText(item.text)}
      {item.children.length > 0 ? (
        <ul>
          {item.children.map((child, index) => (
            <li key={`${child.text}-${index}`}>{renderListItem(child)}</li>
          ))}
        </ul>
      ) : null}
    </>
  );
}

function renderListItemText(item: string): ReactNode {
  const splitItem = splitListItem(item);

  if (!splitItem) {
    return renderInlineText(item);
  }

  return (
    <>
      <strong>{stripMarkdownMarkers(splitItem.title)}</strong>
      {splitItem.detail ? <span>{renderInlineText(splitItem.detail)}</span> : null}
    </>
  );
}

function splitListItem(item: string): { title: string; detail: string } | null {
  const boldTitleMatch = item.match(/^\*\*(.+?)\*\*\s*[:：]\s*(.*)$/);

  if (boldTitleMatch) {
    return { title: boldTitleMatch[1], detail: boldTitleMatch[2] };
  }

  const titleMatch = item.match(/^([^:：]{1,24})[:：]\s*(.+)$/);

  if (!titleMatch) return null;

  return { title: titleMatch[1], detail: titleMatch[2] };
}

function renderInlineText(text: string): ReactNode {
  return text.split(BOLD_SEGMENT_PATTERN).map((segment, index) => {
    if (segment.startsWith("**") && segment.endsWith("**")) {
      return <strong key={`${segment}-${index}`}>{stripMarkdownMarkers(segment)}</strong>;
    }

    return <Fragment key={`${segment}-${index}`}>{segment}</Fragment>;
  });
}

function stripMarkdownMarkers(text: string): string {
  return text.replace(/\*\*/g, "").trim();
}
