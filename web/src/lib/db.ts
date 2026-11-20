import type { Database, SqlJsStatic } from "sql.js";

import { type Column, createStatement } from "./schema";

declare global {
  interface Window {
    initSqlJs?: (config?: { locateFile?: (file: string) => string }) => Promise<SqlJsStatic>;
  }
}

export type Table = { name: string; columns: Column[]; rows: string[][] };
export type QueryResult = { columns: string[]; rows: unknown[][] } | { error: string };

let sqlPromise: Promise<SqlJsStatic> | null = null;

/** The SQLite engine (WebAssembly), loaded once from /public as a plain script. */
export function loadSql(): Promise<SqlJsStatic> {
  if (!sqlPromise) {
    const loading = new Promise<SqlJsStatic>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/sql-wasm.js";
      script.onload = () => {
        const init = window.initSqlJs;
        if (!init) {
          reject(new Error("the SQLite engine did not start"));
          return;
        }
        init({ locateFile: (file) => `/${file}` }).then(resolve, reject);
      };
      script.onerror = () => reject(new Error("could not load the SQLite engine"));
      document.head.appendChild(script);
    });
    loading.catch(() => {
      sqlPromise = null; // allow a retry
    });
    sqlPromise = loading;
  }
  return sqlPromise;
}

function convert(value: string | undefined, type: Column["type"]): string | number | null {
  const text = (value ?? "").trim();
  if (text === "") return null;
  if (type === "INTEGER" || type === "REAL") {
    const number = Number(text);
    return Number.isNaN(number) ? text : number;
  }
  return text;
}

/** An in-memory database holding the CSV as one table. */
export async function buildDatabase(table: Table): Promise<Database> {
  const SQL = await loadSql();
  const db = new SQL.Database();
  db.run(createStatement(table.name, table.columns));
  const marks = table.columns.map(() => "?").join(", ");
  const insert = db.prepare(`INSERT INTO ${table.name} VALUES (${marks})`);
  db.run("BEGIN");
  for (const row of table.rows) {
    insert.run(table.columns.map((column, i) => convert(row[i], column.type)));
  }
  db.run("COMMIT");
  insert.free();
  return db;
}

const READ_ONLY = /^\s*(select|with)\b/i;
const MORE_STATEMENTS = /;\s*\S/;

/** Runs one read-only query and returns its rows, or the error. */
export function runQuery(db: Database, sql: string): QueryResult {
  if (!READ_ONLY.test(sql)) return { error: "only SELECT queries are allowed" };
  if (MORE_STATEMENTS.test(sql)) return { error: "one statement at a time" };
  try {
    const results = db.exec(sql);
    const last = results[results.length - 1];
    return last ? { columns: last.columns, rows: last.values } : { columns: [], rows: [] };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}