export type Example = { question: string; sql: string };

/** Answers the live model returned for the sample singer table. */
export const EXAMPLES: Example[] = [
  {
    question: "How many singers are older than 30?",
    sql: "SELECT COUNT(*) FROM singer WHERE age > 30",
  },
  {
    question: "Show the names of singers from France",
    sql: "SELECT name FROM singer WHERE country = 'France'",
  },
];

function normalize(question: string): string {
  return question.trim().toLowerCase().replace(/[?.!\s]+$/, "");
}

export function findExample(question: string): Example | undefined {
  const key = normalize(question);
  return EXAMPLES.find((e) => normalize(e.question) === key);
}