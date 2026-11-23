export const metadata = { title: "How it works · querion" };

const steps = [
  "A 101M-parameter model, trained from scratch on one laptop GPU, then fine-tuned on text-to-SQL pairs.",
  "Decoding is limited to tokens that fit your table definition. Spider dev: 21.5% to 26.2%.",
  "The server samples several candidate queries: 1, 4 or 8.",
  "Each candidate runs on your data in the browser. Errors and empty results drop out; the majority result wins.",
  "The server never sees your rows. Queries are read-only, one statement, stopped after 5 seconds.",
  "A custom engine serves it: continuous batching, paged KV cache and prefix cache, checked for identical output.",
];

export default function HowItWorks() {
  return (
    <main className="wrap" style={{ paddingBottom: 96 }}>
      <section style={{ padding: "64px 0 48px" }}>
        <h1 className="h1 u">How it works</h1>
      </section>
      {steps.map((text, i) => (
        <div className="k u" key={i} style={{ animationDelay: `${0.2 + i * 0.1}s` }}>
          <span className="mo">{String(i + 1).padStart(2, "0")}</span>
          <span style={{ flex: 1, minWidth: 240 }}>{text}</span>
        </div>
      ))}
    </main>
  );
}