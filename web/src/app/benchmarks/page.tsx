import DataTable from "@/components/DataTable";
import data from "@/data/benchmarks.json";

export const metadata = { title: "Benchmarks · QUERION" };

const percent = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function Benchmarks() {
  const { accuracy, taxonomy, engine, pipeline } = data;
  const failures = Object.entries(taxonomy.labels as Record<string, number>).sort(
    (a, b) => b[1] - a[1],
  );
  return (
    <main className="mx-auto max-w-4xl space-y-8 p-6">
      <header>
        <h1 className="text-3xl font-semibold">Benchmarks</h1>
        <p className="opacity-70">
          Every number here is generated from the JSON files in the repository&apos;s
          benchmarks folder. Nothing is typed in by hand.
        </p>
      </header>

      <section className="space-y-2">
        <h2 className="text-xl font-medium">Accuracy on Spider dev</h2>
        <p className="text-sm opacity-70">
          {accuracy.benchmark}. This is not Spider&apos;s official test-suite score. Each row
          adds one step. Standard error is the sampling error over {accuracy.runs[0].items}{" "}
          questions.
        </p>
        <DataTable
          headers={["Step", "Matches", "Accuracy", "Std. error"]}
          rows={accuracy.runs.map((r) => [
            r.label,
            `${r.matches} of ${r.items}`,
            percent(r.accuracy),
            `±${percent(r.stdError)}`,
          ])}
        />
        <p className="text-sm opacity-70">
          Repeated vote runs through the serving engine landed between 34.6% and 35.6%, so
          the vote row is good to about a point.
        </p>
      </section>

      <section className="space-y-2">
        <h2 className="text-xl font-medium">Where the vote run still fails</h2>
        <p className="text-sm opacity-70">
          {taxonomy.failures} failures out of {taxonomy.items} questions, sorted by a
          rule-based labeller. A hand check of 40 labels found 30 right, 8 right but coarse
          and 2 wrong.
        </p>
        <DataTable
          headers={["Failure type", "Count"]}
          rows={failures.map(([label, count]) => [label, String(count)])}
        />
      </section>

      <section className="space-y-2">
        <h2 className="text-xl font-medium">Full pipeline through the engine</h2>
        <DataTable headers={pipeline.headers} rows={pipeline.rows} />
      </section>

      <section className="space-y-2">
        <h2 className="text-xl font-medium">Serving engine, interleaved A/B runs</h2>
        <p className="text-sm opacity-70">
          The laptop&apos;s absolute speed varies by 25 to 40% between sessions, so engines
          are compared back to back and only the ratios are meaningful.
        </p>
        <DataTable headers={engine.headers} rows={engine.rows} />
      </section>
    </main>
  );
}