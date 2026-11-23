import DataTable from "@/components/DataTable";

export const metadata = { title: "Model card · QUERION" };

const facts = [
  ["Parameters", "101,026,560"],
  ["Architecture", "12 layers, d_model 768, 12 query heads, 4 KV heads, context 2048"],
  ["Tokenizer", "24k byte-level BPE trained on SQL and text"],
  ["Pretraining", "2B tokens: 59% FineWeb-Edu, 40% Stack SQL, 1% pairs"],
  ["Fine-tuning", "Text-to-SQL pairs with identifier augmentation, loss on SQL tokens only"],
  ["Hardware", "One laptop RTX 4050, 6 GB"],
  ["License", "Not yet set. Training data licenses are still being checked."],
];

export default function ModelCard() {
  return (
    <main className="mx-auto max-w-4xl space-y-6 p-6">
      <h1 className="text-3xl font-semibold">Model card</h1>
      <DataTable headers={["", ""]} rows={facts} />
      <section className="space-y-2 text-sm">
        <h2 className="text-xl font-medium">Limits</h2>
        <p className="opacity-80">
          It is a small model. On Spider dev it answers about 35% of questions correctly
          with the full pipeline. It picks the wrong tables or joins most often. Check
          the SQL before you rely on a result, and use the Wrong answer button when it
          fails.
        </p>
      </section>
      <p className="text-sm">
        <a className="underline" href="https://huggingface.co/SinghBro/querion-100m-sql">
          Weights on Hugging Face
        </a>
        {" · "}
        <a className="underline" href="https://github.com/AmanSingh-404/QUERION">
          Source on GitHub
        </a>
      </p>
    </main>
  );
}