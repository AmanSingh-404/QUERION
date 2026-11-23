import Link from "next/link";

import data from "@/data/benchmarks.json";

const steps = ["Pretrained, zero-shot", "+ fine-tuning", "+ identifier augmentation", "+ schema-constrained decoding", "+ 8 samples, execution vote"];
const phrases = ["Trained from scratch", "Runs on your data", "Checked by execution", "One laptop GPU"];
const principles = [
  ["Private.", "The CSV lives in a SQLite database inside your browser."],
  ["Checked.", "Every candidate runs on your data. The majority result wins."],
  ["Fast.", "Own serving engine: continuous batching, paged and prefix cache."],
];

export default function Home() {
  const runs = data.accuracy.runs;
  return (
    <main className="wrap">
      <section className="hero">
        <div className="dots" aria-hidden="true" />
        <p className="mo m u">Text-to-SQL · 101M parameters</p>
        <h1 className="h1">
          <span className="w">
            <span style={{ animationDelay: ".15s" }}>Ask your data</span>
          </span>
          <span className="w">
            <span style={{ animationDelay: ".3s", color: "var(--orange)" }}>in plain English.</span>
          </span>
        </h1>
        <p className="m u lead" style={{ animationDelay: ".35s" }}>
          A small model, trained from scratch, that writes SQL for your CSV. Your rows never leave the browser.
        </p>
        <div className="u row-actions" style={{ animationDelay: ".5s" }}>
          <Link className="btn" href="/playground">
            Open playground
          </Link>
          <Link className="lk" href="/benchmarks">
            Benchmarks
          </Link>
        </div>
      </section>

      <section className="demo u" style={{ animationDelay: ".7s" }}>
        <p className="mo m">How many singers are older than 30?</p>
        <p className="mo sqlline">
          <span className="ty">SELECT COUNT(*) FROM singer WHERE age &gt; 30</span>
        </p>
        <p className="big res">4</p>
      </section>

      <section className="marquee" aria-hidden="true">
        <div>
          {[...phrases, ...phrases].map((text, i) => (
            <span key={i} className={i % 2 ? "m" : ""}>
              {text}
            </span>
          ))}
        </div>
      </section>

      <section className="stats">
        <div>
          <p className="n1 big" aria-label="101 million parameters" />
          <p className="m" style={{ marginTop: 12 }}>parameters, trained from scratch</p>
        </div>
        <div>
          <p className="n2 big" aria-label="35.3 percent" />
          <p className="m" style={{ marginTop: 12 }}>Spider dev, 8 samples and a vote</p>
        </div>
        <div>
          <p className="n3 big" aria-label="5.4 times" />
          <p className="m" style={{ marginTop: 12 }}>throughput at 16 clients</p>
        </div>
      </section>

      <section className="sec">
        <h2 className="h2" style={{ marginBottom: 40 }}>Every step, measured.</h2>
        {runs.map((run, i) => (
          <div className="r" key={run.label}>
            <span>{steps[i]}</span>
            <div className="trk">
              <div className="bar" style={{ width: `${Math.round((run.accuracy / 0.4) * 100)}%`, animationDelay: `${i * 0.15}s` }} />
            </div>
            <span>{(run.accuracy * 100).toFixed(1)}%</span>
          </div>
        ))}
        <p className="m" style={{ fontSize: 13, marginTop: 16 }}>
          Execution accuracy, {runs[0].items.toLocaleString()} Spider dev questions. Not the official score.
        </p>
      </section>

      <section style={{ padding: "48px 0" }}>
        {principles.map(([title, text]) => (
          <div className="pr" key={title}>
            <h3 className="h2">{title}</h3>
            <p>{text}</p>
          </div>
        ))}
      </section>

      <section className="cta">
        <h2>Try it.</h2>
        <Link className="btn" href="/playground">
          Open playground
        </Link>
      </section>

      <div className="mark" aria-hidden="true">
        <p>querion</p>
      </div>
    </main>
  );
}