"use client";

import Papa from "papaparse";
import { useRef, useState } from "react";

import { TUNNEL_HEADERS, generate } from "@/lib/api";
import { API_URL } from "@/lib/config";
import { EXAMPLES, findExample } from "@/lib/examples";
import { createStatement, inferColumns, safeName } from "@/lib/schema";
import { type QueryResult, SqlSession, type Table } from "@/lib/sqlsession";
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

function ResultTable({ result, limit = 100, total }: { result: QueryResult; limit?: number; total?: number }) {
  if ("error" in result) return <p style={{ color: "#b42318", fontSize: 14 }}>{result.error}</p>;
  const shown = result.rows.slice(0, limit);
  const count = total ?? result.rows.length;
  return (
    <div className="sc">
      <table>
        <thead>
          <tr>
            {result.columns.map((column, i) => (
              <th key={i}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row, i) => (
            <tr key={i}>
              {row.map((value, j) => (
                <td key={j}>{value === null ? "NULL" : String(value)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="m" style={{ fontSize: 12, marginTop: 8 }}>
        {count} rows{count > shown.length ? ` (first ${shown.length} shown)` : ""}
      </p>
    </div>
  );
}

export default function Playground() {
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
  const [recorded, setRecorded] = useState(false);
  const sessionRef = useRef<SqlSession | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  function clearAnswer() {
    setError("");
    setCandidates([]);
    setVote(null);
    setSql("");
    setResult(null);
    setRated("");
    setRecorded(false);
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
      const session = sessionRef.current ?? new SqlSession();
      await session.load(next);
      sessionRef.current = session;
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
    const session = sessionRef.current;
    if (!table || !session || !question.trim() || busy) return;
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
          const outcome = await session.run(event.sql);
          const entry = { index: event.index, sql: event.sql, result: outcome };
          setCandidates((previous) => [...previous, entry]);
          setStatus("Candidates are arriving...");
        } else if (event.type === "done") {
          final = event.candidates;
        }
      }
      setStatus("Voting...");
      const picked = await chooseCandidate(final, (query) => session.run(query));
      if (!picked) {
        setError("The model did not return a query. Try rephrasing the question.");
      } else {
        setVote(picked);
        setSql(picked.sql);
        setResult(await session.run(picked.sql));
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      const known = table.name === "singer" ? findExample(question) : undefined;
      if (known) {
        setRecorded(true);
        setVote({ index: 0, sql: known.sql, agree: 0, ran: 0 });
        setSql(known.sql);
        setResult(await session.run(known.sql));
      } else {
        setError(friendly(e));
      }
    } finally {
      if (abortRef.current === controller) {
        setBusy(false);
        setStatus("");
      }
    }
  }

  async function runEdited() {
    const session = sessionRef.current;
    if (session) setResult(await session.run(sql));
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
        headers: { "content-type": "application/json", ...TUNNEL_HEADERS },
        body: JSON.stringify({ question, sql, verdict }),
      });
    } catch {
      /* feedback is best effort */
    }
  }

  const grow = { flex: 1, minWidth: 260 } as const;
  return (
    <main className="wrap" style={{ paddingBottom: 96 }}>
      <section style={{ padding: "64px 0 48px" }}>
        <h1 className="h1 u">Playground</h1>
        <p className="m u" style={{ animationDelay: ".2s" }}>
          Only the table definition and your question leave the browser.
        </p>
      </section>

      <section className="k u" style={{ animationDelay: ".3s" }}>
        <span className="mo">01 Data</span>
        <div style={grow}>
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 28, marginBottom: 20 }}>
            <input
              className="mo"
              type="file"
              accept=".csv,text/csv"
              aria-label="Choose a CSV file"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onFile(file);
              }}
            />
            <button type="button" className="lk" onClick={loadSample}>
              Use sample data
            </button>
          </div>
          {fileError && <p style={{ color: "#b42318", fontSize: 14 }}>{fileError}</p>}
          {table && (
            <>
              <p className="mo m" style={{ marginBottom: 12 }}>
                {table.name}({table.columns.map((c) => `${c.name} ${c.type}`).join(", ")})
              </p>
              <ResultTable
                result={{ columns: table.columns.map((c) => c.name), rows: table.rows.slice(0, 5) }}
                limit={5}
                total={table.rows.length}
              />
            </>
          )}
        </div>
      </section>

      {table && (
        <section className="k u">
          <span className="mo">02 Question</span>
          <div style={grow}>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 24 }}>
              <input
                className="in"
                value={question}
                placeholder="How many singers are older than 30?"
                aria-label="Your question"
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void ask();
                }}
              />
              <select
                className="in"
                style={{ flex: "0 0 auto", minWidth: 0, fontSize: 16 }}
                value={samples}
                aria-label="Number of candidates"
                onChange={(e) => setSamples(Number(e.target.value))}
              >
                <option value={1}>1 candidate</option>
                <option value={4}>4 candidates</option>
                <option value={8}>8 candidates</option>
              </select>
              <button type="button" className="btn" disabled={busy} onClick={() => void ask()}>
                Ask
              </button>
            </div>
            {table.name === "singer" && (
              <p className="mo m" style={{ marginTop: 16, display: "flex", flexWrap: "wrap", gap: "8px 24px" }}>
                Try:
                {EXAMPLES.map((example) => (
                  <button
                    key={example.question}
                    type="button"
                    className="lk mo"
                    onClick={() => setQuestion(example.question)}
                  >
                    {example.question}
                  </button>
                ))}
              </p>
            )}
          </div>
        </section>
      )}

      {(busy || error || candidates.length > 0 || vote) && (
        <section className="k u">
          <span className="mo">03 Answer</span>
          <div style={grow}>
            {error && <p style={{ color: "#b42318", fontSize: 14 }}>{error}</p>}
            {busy && <p className="m">{status}</p>}
            {candidates.length > 0 && (
              <details open={!vote} style={{ marginBottom: 16 }}>
                <summary className="m" style={{ cursor: "pointer", fontSize: 14 }}>
                  Candidates ({candidates.length})
                </summary>
                <ul style={{ listStyle: "none", padding: 0, margin: "12px 0 0" }}>
                  {candidates.map((c) => (
                    <li key={c.index} className="mo" style={{ padding: "8px 0", fontWeight: vote?.index === c.index ? 500 : 400 }}>
                      <span style={{ color: "var(--orange)" }}>{vote?.index === c.index ? "● " : "○ "}</span>
                      <span style={{ wordBreak: "break-all" }}>{c.sql}</span>
                      <span className="m"> · {"error" in c.result ? "error" : `${c.result.rows.length} rows`}</span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {vote && result && (
              <div>
                <p className="m" style={{ fontSize: 14, marginBottom: 12 }}>
                  {recorded
                    ? "The live model is offline. This is a recorded answer the model gave for this exact question."
                    : vote.ran === 0
                      ? "None of the candidates ran, so this is the first one."
                      : `${vote.agree} of ${vote.ran} runnable candidates agree.`}
                </p>
                <textarea
                  className="mo"
                  style={{ width: "100%", boxSizing: "border-box", padding: "12px 0", border: 0, borderBottom: "2px solid var(--orange)", background: "transparent", fontSize: 16, outline: "none", resize: "vertical" }}
                  rows={2}
                  value={sql}
                  aria-label="SQL"
                  onChange={(e) => setSql(e.target.value)}
                />
                <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 28, margin: "20px 0" }}>
                  <button type="button" className="lk" onClick={() => void runEdited()}>
                    Run
                  </button>
                  <button type="button" className="lk" onClick={exportCsv}>
                    Export CSV
                  </button>
                  <button type="button" className="btn-good" disabled={!!rated} onClick={() => void rate("up")}>
                    Good answer
                  </button>
                  <button type="button" className="btn-bad" disabled={!!rated} onClick={() => void rate("down")}>
                    Wrong answer
                  </button>
                  {rated && <span className="m" style={{ fontSize: 14 }}>Thanks, that helps.</span>}
                </div>
                <ResultTable result={result} />
              </div>
            )}
          </div>
        </section>
      )}
    </main>
  );
}