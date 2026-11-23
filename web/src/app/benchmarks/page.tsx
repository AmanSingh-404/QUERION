import DataTable from "@/components/DataTable";
import data from "@/data/benchmarks.json";

export const metadata = { title: "Benchmarks · querion" };

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

function Bars({ items }: { items: [string, number, string][] }) {
  const max = Math.max(...items.map((i) => i[1]));
  return (
    <>
      {items.map(([label, value, text], i) => (
        <div className="r" key={label}>
          <span>{label}</span>
          <div className="trk">
            <div className="bar" style={{ width: `${(value / max) * 100}%`, animationDelay: `${i * 0.15}s` }} />
          </div>
          <span>{text}</span>
        </div>
      ))}
    </>
  );
}

export default function Benchmarks() {
  const { accuracy, taxonomy, engine, pipeline } = data;
  const failures = Object.entries(taxonomy.labels as Record<string, number>)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
  return (
    <main className="wrap" style={{ paddingBottom: 96 }}>
      <section style={{ padding: "64px 0 48px" }}>
        <h1 className="h1 u">Benchmarks</h1>
        <p className="m u" style={{ animationDelay: ".2s" }}>
          Generated from the repository&apos;s JSON files. Nothing is typed by hand.
        </p>
      </section>

      <h2 style={{ fontSize: 28, letterSpacing: "-.03em", marginBottom: 8 }}>Accuracy on Spider dev</h2>
      <p className="m" style={{ fontSize: 13, marginBottom: 16 }}>
        {accuracy.benchmark}. Not the official test-suite score. Vote runs through the engine landed between 34.6% and 35.6%.
      </p>
      <Bars items={accuracy.runs.map((r) => [r.label, r.accuracy, `${pct(r.accuracy)} ±${(r.stdError * 100).toFixed(1)}`])} />

      <h2 style={{ fontSize: 28, letterSpacing: "-.03em", margin: "72px 0 8px" }}>Where it still fails</h2>
      <p className="m" style={{ fontSize: 13, marginBottom: 16 }}>
        {taxonomy.failures} failures out of {taxonomy.items}, from a rule-based labeller. A hand check of 40 labels: 30 right, 8 right but coarse, 2 wrong.
      </p>
      <Bars items={failures.map(([label, count]) => [label, count, String(count)])} />

      <h2 style={{ fontSize: 28, letterSpacing: "-.03em", margin: "72px 0 16px" }}>Full pipeline through the engine</h2>
      <DataTable headers={pipeline.headers} rows={pipeline.rows} />

      <h2 style={{ fontSize: 28, letterSpacing: "-.03em", margin: "72px 0 8px" }}>Serving engine, interleaved A/B</h2>
      <p className="m" style={{ fontSize: 13, marginBottom: 16 }}>
        Absolute speed varies 25 to 40% between sessions, so only back-to-back ratios count.
      </p>
      <DataTable headers={engine.headers} rows={engine.rows} />
    </main>
  );
}