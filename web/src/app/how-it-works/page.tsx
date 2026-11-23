export const metadata = { title: "How it works · QUERION" };

const steps = [
  [
    "The model",
    "A 101M-parameter decoder-only transformer (12 layers, 4 KV heads, RoPE, SwiGLU) trained from scratch on one laptop GPU: 2B tokens of pretraining, then supervised fine-tuning on text-to-SQL pairs.",
  ],
  [
    "Schema-constrained decoding",
    "While generating, only tokens that keep the query valid against your table definition are allowed. This took Spider dev from 21.5% to 26.2%.",
  ],
  [
    "Several candidates",
    "The server samples several queries for your question (1, 4 or 8 in the playground).",
  ],
  [
    "Execution-guided vote, in your browser",
    "Each candidate runs against your data in a SQLite database inside a Web Worker. Candidates that fail or return nothing are dropped and the result most candidates agree on wins. Ties go to the earliest. Eight samples took Spider dev to about 35%.",
  ],
  [
    "Your data stays with you",
    "Your CSV is loaded into the browser only. The server receives the table definition and your question, never the rows. Queries are read-only, one statement at a time, and stopped after 5 seconds.",
  ],
  [
    "The serving engine",
    "Continuous batching, a paged KV cache and a prefix cache, all tested to give byte-identical output to the simple server. See the benchmarks page for the measured effect of each.",
  ],
];

export default function HowItWorks() {
  return (
    <main className="mx-auto max-w-4xl space-y-6 p-6">
      <h1 className="text-3xl font-semibold">How it works</h1>
      <ol className="space-y-4">
        {steps.map(([title, text], i) => (
          <li key={title} className="rounded border border-black/10 p-4 dark:border-white/20">
            <h2 className="font-medium">
              {i + 1}. {title}
            </h2>
            <p className="mt-1 text-sm opacity-80">{text}</p>
          </li>
        ))}
      </ol>
    </main>
  );
}