"use client";

import Papa from "papaparse";
import { useRef, useState } from "react";
import type { Database } from "sql.js";

import { generate } from "@/lib/api";
import { API_URL } from "@/lib/config";
import { type QueryResult, type Table, buildDatabase, runQuery } from "@/lib/db";
import { createStatement, inferColumns, safeName } from "@/lib/schema";
import { type Vote, chooseCandidate } from "@/lib/vote";

const MAX_BYTES = 5 * 1024 * 1024;
const MAX_ROWS = 20000;
const MAX_COLUMNS = 40;
const SAMPLE_CSV = [
  "singer_id,name,country,age",
  "1,Joe Sharp,Netherlands,52",
  "2,Timbaland,United States,32",
  "3,Justin Brown,France,29",
  "4,Rose White,France,41",
  "5,John Nizinik,France,43",
  "6,Tribal King,France,25",
].join("\n");

type Candidate = { index: number; sql: string; result: QueryResult };

function friendly(e: unknown): string {
  const message = e instanceof Error ? e.message : String(e);
  if (/failed to fetch|networkerror|load failed/i.test(message)) {
    return "Cannot reach the model server. If it was idle it may need a minute to wake up; try again shortly.";
  }
  return message;
}

function ResultTable({
  result,
  limit = 100,
  total,
}: {
  result: QueryResult;
  limit?: number;
  total?: number;
}) {
  if ("error" in result) return <p className="text-sm text-red-600">{result.error}</p>;
  const shown = result.rows.slice(0, limit);
  const count = total ?? result.rows.length;
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-sm">
        <thead>
          <tr>
            {result.columns.map((column, i) => (
              <th key={i} className="border-b px-2 py-1 font-medium">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row, i) => (
            <tr key={i}>
              {row.map((value, j) => (
                <td key={j} className="border-b px-2 py-1">
                  {value === null ? "NULL" : String(value)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-xs opacity-70">
        {count} rows{count > shown.length ? ` (first ${shown.length} shown)` : ""}
      </p>
    </div>
  );
}

const button =
  "rounded bg-black px-4 py-2 text-sm text-white disabled:opacity-50 dark:bg-white dark:text-black";
const card = "space-y-3 rounded border border-black/10 p-4 dark:border-white/20";

export default function Home() {
  const [table, setTable] = useState<Table | null>(null);
  const [fileError, setFileError] = useState("");
  const [question, setQuestion] = useState("");
  const [samples, setSamples] = useState(4);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [vote, setVote] = useState<Vote | null>(null);
  const [sql, setSql] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [rated, setRated] = useState("");
  const dbRef = useRef<Database | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  function clearAnswer() {
    setError("");
    setCandidates([]);
    setVote(null);
    setSql("");
    setResult(null);
    setRated("");
  }

  async function prepare(data: string[][], fileName: string) {
    setFileError("");
    if (data.length < 2) {
      setFileError("The file needs a header row and at least one data row.");
      return;
    }
    const [header, ...rows] = data;
    if (header.length > MAX_COLUMNS) {
      setFileError(`At most ${MAX_COLUMNS} columns in this demo.`);
      return;
    }
    if (rows.length > MAX_ROWS) {
      setFileError(`At most ${MAX_ROWS.toLocaleString()} rows in this demo.`);
      return;
    }
    try {
      const name = safeName(fileName.replace(/\.[^.]+$/, ""), "data");
      const next: Table = { name, columns: inferColumns(header, rows), rows };
      const db = await buildDatabase(next);
      dbRef.current?.close();
      dbRef.current = db;
      setTable(next);
      clearAnswer();
    } catch (e) {
      setFileError(e instanceof Error ? e.message : "Could not read that file.");
    }
  }

  function onFile(file: File) {
    if (file.size > MAX_BYTES) {
      setFileError("That file is larger than 5 MB.");
      return;
    }
    Papa.parse<string[]>(file, {
      skipEmptyLines: true,
      complete: (parsed) => void prepare(parsed.data, file.name),
    });
  }

  function loadSample() {
    const parsed = Papa.parse<string[]>(SAMPLE_CSV, { skipEmptyLines: true });
    void prepare(parsed.data, "singer.csv");
  }

  async function ask() {
    const db = dbRef.current;
    if (!table || !db || !question.trim() || busy) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    clearAnswer();
    setBusy(true);
    setStatus("Sending the question...");
    try {
      const body = {
        tables: [createStatement(table.name, table.columns)],
        question: question.trim(),
        samples,
      };
      let final: (string | null)[] = [];
      for await (const event of generate(API_URL, body, controller.signal)) {
        if (event.type === "candidate") {
          const entry = { index: event.index, sql: event.sql, result: runQuery(db, event.sql) };
          setCandidates((previous) => [...previous, entry]);
          setStatus("Candidates are arriving...");
        } else if (event.type === "done") {
          final = event.candidates;
        }
      }
      setStatus("Voting...");
      const picked = chooseCandidate(final, (query) => runQuery(db, query));
      if (!picked) {
        setError("The model did not return a query. Try rephrasing the question.");
      } else {
        setVote(picked);
        setSql(picked.sql);
        setResult(runQuery(db, picked.sql));
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(friendly(e));
    } finally {
      if (abortRef.current === controller) {
        setBusy(false);
        setStatus("");
      }
    }
  }

  function runEdited() {
    const db = dbRef.current;
    if (db) setResult(runQuery(db, sql));
  }

  function exportCsv() {
    if (!result || "error" in result) return;
    const csv = Papa.unparse({ fields: result.columns, data: result.rows });
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "result.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  async function rate(verdict: "up" | "down") {
    setRated(verdict);
    try {
      await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question, sql, verdict }),
      });
    } catch {
      /* feedback is best effort */
    }
  }

  return (
    <main className="mx-auto max-w-4xl space-y-6 p-6">
      <header>
        <h1 className="text-3xl font-semibold">QUERION</h1>
        <p className="opacity-70">
          Ask your data in plain English. Your data stays in your browser: only the table
          definition and your question are sent to the model.
        </p>
      </header>

      <section className={card}>
        <h2 className="font-medium">1. Your data</h2>
        <div className="flex flex-wrap items-center gap-3">
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFile(file);
            }}
          />
          <button type="button" className={button} onClick={loadSample}>
            Use sample data
          </button>
        </div>
        {fileError && <p className="text-sm text-red-600">{fileError}</p>}
        {table && (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2">
              <span className="text-sm font-medium">{table.name}</span>
              {table.columns.map((column) => (
                <span key={column.name} className="rounded-full border px-2 py-0.5 text-xs">
                  {column.name}: {column.type}
                </span>
              ))}
            </div>
            <ResultTable
              result={{ columns: table.columns.map((c) => c.name), rows: table.rows.slice(0, 5) }}
              limit={5}
              total={table.rows.length}
            />
          </div>
        )}
      </section>

      {table && (
        <section className={card}>
          <h2 className="font-medium">2. Your question</h2>
          <div className="flex flex-wrap gap-3">
            <input
              className="min-w-64 flex-1 rounded border p-2"
              value={question}
              placeholder="e.g. How many singers are older than 30?"
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void ask();
              }}
            />
            <select
              className="rounded border p-2 text-sm"
              value={samples}
              onChange={(e) => setSamples(Number(e.target.value))}
            >
              <option value={1}>Fast (1 candidate)</option>
              <option value={4}>Balanced (4 candidates)</option>
              <option value={8}>Best (8 candidates)</option>
            </select>
            <button type="button" className={button} disabled={busy} onClick={() => void ask()}>
              Ask
            </button>
          </div>
        </section>
      )}

      {(busy || error || candidates.length > 0 || vote) && (
        <section className={card}>
          <h2 className="font-medium">3. Answer</h2>
          {error && <p className="text-sm text-red-600">{error}</p>}
          {busy && <p className="text-sm opacity-70">{status}</p>}
          {candidates.length > 0 && (
            <details open={!vote}>
              <summary className="cursor-pointer text-sm font-medium">
                Candidates ({candidates.length})
              </summary>
              <ul className="mt-2 space-y-1">
                {candidates.map((c) => (
                  <li
                    key={c.index}
                    className={`rounded p-2 text-sm ${vote?.index === c.index ? "bg-green-500/10" : ""}`}
                  >
                    <code className="break-all">{c.sql}</code>
                    <span className="ml-2 text-xs opacity-70">
                      {"error" in c.result ? "error" : `${c.result.rows.length} rows`}
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}
          {vote && result && (
            <div className="space-y-3">
              <p className="text-sm opacity-70">
                {vote.ran === 0
                  ? "None of the candidates ran, so this is the first one."
                  : `${vote.agree} of ${vote.ran} runnable candidates returned this result.`}
              </p>
              <textarea
                className="w-full rounded border p-2 font-mono text-sm"
                rows={3}
                value={sql}
                onChange={(e) => setSql(e.target.value)}
              />
              <div className="flex flex-wrap gap-2">
                <button type="button" className={button} onClick={runEdited}>
                  Run
                </button>
                <button type="button" className={button} onClick={exportCsv}>
                  Export CSV
                </button>
                <button
                  type="button"
                  className={button}
                  disabled={!!rated}
                  onClick={() => void rate("up")}
                >
                  Good answer
                </button>
                <button
                  type="button"
                  className={button}
                  disabled={!!rated}
                  onClick={() => void rate("down")}
                >
                  Wrong answer
                </button>
              </div>
              <ResultTable result={result} />
            </div>
          )}
        </section>
      )}
    </main>
  );
}