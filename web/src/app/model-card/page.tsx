export const metadata = { title: "Model · querion" };

const facts = [
  ["Parameters", "101,026,560"],
  ["Architecture", "12 layers, d_model 768, 12 query heads, 4 KV heads, context 2048"],
  ["Tokenizer", "24k byte-level BPE trained on SQL and text"],
  ["Pretraining", "2B tokens: 59% FineWeb-Edu, 40% Stack SQL, 1% pairs"],
  ["Fine-tuning", "Text-to-SQL pairs with identifier augmentation"],
  ["Hardware", "One laptop RTX 4050, 6 GB"],
  ["License", "Code: MIT. Weights: license not yet chosen while the training data licenses are checked."],
];

export default function ModelCard() {
  return (
    <main className="wrap" style={{ paddingBottom: 96 }}>
      <section style={{ padding: "64px 0 48px" }}>
        <h1 className="h1 u">Model</h1>
      </section>
      {facts.map(([label, value], i) => (
        <div className="k u" key={label} style={{ animationDelay: `${0.2 + i * 0.08}s` }}>
          <span>{label}</span>
          <span style={{ flex: 1, minWidth: 240 }}>{value}</span>
        </div>
      ))}
      <p className="m" style={{ marginTop: 32, maxWidth: 560 }}>
        A small model: about 35% on Spider dev with the full pipeline. It most often picks the wrong tables or joins.
        Check the SQL before relying on a result, and press Wrong answer when it fails.
      </p>
      <p style={{ marginTop: 32, display: "flex", gap: 32, flexWrap: "wrap" }}>
        <a className="lk" href="https://huggingface.co/SinghBro/querion-100m-sql">
          Weights on Hugging Face
        </a>
        <a className="lk" href="https://github.com/AmanSingh-404/QUERION">
          Source on GitHub
        </a>
      </p>
    </main>
  );
}