export type Column = { name: string; type: "INTEGER" | "REAL" | "TEXT" };

const RESERVED = new Set([
  "select", "from", "where", "group", "order", "by", "having", "limit", "offset", "join",
  "on", "as", "and", "or", "not", "in", "is", "null", "like", "between", "table", "index",
  "key", "values", "union", "distinct", "case", "when", "then", "else", "end", "all",
  "create", "insert", "update", "delete", "drop",
]);

/** A name that is safe to use unquoted in SQL. */
export function safeName(raw: string, fallback: string): string {
  const cleaned = raw
    .trim()
    .replace(/[^A-Za-z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!cleaned) return fallback;
  const name = /^[0-9]/.test(cleaned) ? `c_${cleaned}` : cleaned;
  return RESERVED.has(name.toLowerCase()) ? `${name}_` : name;
}

/** Column names and types from a CSV header and its rows (the first `sample` rows decide). */
export function inferColumns(header: string[], rows: string[][], sample = 200): Column[] {
  const used = new Set<string>();
  return header.map((raw, i) => {
    let name = safeName(raw, `column_${i + 1}`);
    while (used.has(name.toLowerCase())) name += "_x";
    used.add(name.toLowerCase());
    const values = rows
      .slice(0, sample)
      .map((r) => (r[i] ?? "").trim())
      .filter((v) => v !== "");
    let type: Column["type"] = "TEXT";
    if (values.length > 0 && values.every((v) => /^-?\d+$/.test(v))) {
      type = "INTEGER";
    } else if (
      values.length > 0 &&
      values.every((v) => /^-?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(v))
    ) {
      type = "REAL";
    }
    return { name, type };
  });
}

export function createStatement(table: string, columns: Column[]): string {
  return `CREATE TABLE ${table} (${columns.map((c) => `${c.name} ${c.type}`).join(", ")})`;
}