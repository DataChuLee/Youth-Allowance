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
        aria-label="청년수당 질문"
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        placeholder="청년수당에 대해 질문하세요"
        value={value}
      />
      <button disabled={disabled} type="submit">
        전송
      </button>
    </form>
  );
}
