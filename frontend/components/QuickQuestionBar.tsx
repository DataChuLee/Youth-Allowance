const QUESTIONS = [
  "청년수당 사용처 알려줘",
  "활동기록서는 어떻게 제출해?",
  "현금 사용 가능해?",
];

export function QuickQuestionBar({
  disabled,
  onSelect,
}: {
  disabled: boolean;
  onSelect: (question: string) => void;
}) {
  return (
    <div className="quick-question-bar" aria-label="빠른 질문">
      {QUESTIONS.map((question) => (
        <button disabled={disabled} key={question} onClick={() => onSelect(question)} type="button">
          {question}
        </button>
      ))}
    </div>
  );
}
