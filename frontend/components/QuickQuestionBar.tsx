const QUESTIONS = [
  "청년수당 사용처",
  "활동기록서 제출",
  "자격상실 또는 참여 중단",
  "현금 사용 가능 여부",
];

export function QuickQuestionBar({ onSelect }: { onSelect: (question: string) => void }) {
  return (
    <div className="quick-question-bar">
      {QUESTIONS.map((question) => (
        <button key={question} type="button" onClick={() => onSelect(question)}>
          {question}
        </button>
      ))}
    </div>
  );
}
