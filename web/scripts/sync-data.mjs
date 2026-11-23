import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const read = (name) => readFileSync(join(root, "benchmarks", name), "utf8");

function table(markdown) {
  const lines = markdown.split(/\r?\n/).filter((line) => line.trim().startsWith("|"));
  const cells = (line) =>
    line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
  return { headers: cells(lines[0]), rows: lines.slice(2).map(cells) };
}

const accuracy = JSON.parse(read("accuracy.json"));
const taxonomy = JSON.parse(read("error_taxonomy.json"));

const data = {
  accuracy: {
    benchmark: accuracy.benchmark,
    commit: accuracy.collected_at_commit,
    runs: accuracy.runs.map((r) => ({
      label: r.label,
      items: r.items,
      matches: r.matches,
      accuracy: r.exec_accuracy,
      stdError: r.std_error,
    })),
  },
  taxonomy: { failures: taxonomy.failures, items: taxonomy.items, labels: taxonomy.labels },
  engine: table(read("engine_table.md")),
  pipeline: table(read("pipeline_table.md")),
};

const out = join(root, "web", "src", "data");
mkdirSync(out, { recursive: true });
writeFileSync(join(out, "benchmarks.json"), JSON.stringify(data, null, 2) + "\n");
console.log("wrote web/src/data/benchmarks.json");