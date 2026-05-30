import { FormEvent, useState } from "react";

export function ChatInput({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (question: string) => void;
}) {
  const [value, setValue] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const question = value.trim();
    if (!question) return;
    onSubmit(question);
    setValue("");
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        id="question-input"
        aria-label="청년수당 질문"
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        placeholder="원하는 청년수당 문의 내용을 입력하세요. 예: 현금 사용이 가능한가요?"
        value={value}
      />
      <button disabled={disabled} type="submit">
        보내기
      </button>
    </form>
  );
}
