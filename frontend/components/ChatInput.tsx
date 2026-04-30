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
        value={value}
        disabled={disabled}
        placeholder="청년수당에 대해 질문하세요"
        onChange={(event) => setValue(event.target.value)}
      />
      <button disabled={disabled} type="submit">
        전송
      </button>
    </form>
  );
}
