import type { Column } from "./schema";

const READ_ONLY = /^\s*(select|with)\b/i;
const MORE_STATEMENTS = /;\s*\S/;

export type Cell = string | number | null;

/** Why a query may not run here, or null when it is a single read-only statement. */
export function checkSql(sql: string): string | null {
  if (!READ_ONLY.test(sql)) return "only SELECT queries are allowed";
  if (MORE_STATEMENTS.test(sql)) return "one statement at a time";
  return null;
}

/** A CSV cell as a database value: empty becomes NULL, numbers stay numbers. */
export function convertCell(value: string | undefined, type: Column["type"]): Cell {
  const text = (value ?? "").trim();
  if (text === "") return null;
  if (type === "INTEGER" || type === "REAL") {
    const number = Number(text);
    return Number.isNaN(number) ? text : number;
  }
  return text;
}

export function insertStatement(name: string, columns: number): string {
  return `INSERT INTO ${name} VALUES (${new Array(columns).fill("?").join(", ")})`;
}