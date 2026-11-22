export type RunResult = { rows: unknown[][] } | { error: string };
export type Runner = (sql: string) => Promise<RunResult>;
export type Vote = { index: number; sql: string; agree: number; ran: number };

function cell(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : String(Math.round(value * 1e6) / 1e6);
  }
  if (value === null || value === undefined) return "null";
  return "s:" + String(value);
}

/** A canonical form of a result: the same rows in any order give the same string. */
export function signature(rows: unknown[][]): string {
  return rows
    .map((row) => row.map(cell).join("\u0001"))
    .sort()
    .join("\u0002");
}

/**
 * The execution-guided vote, the same rule as the Python version: drop candidates that fail or
 * return nothing, take the result most candidates agree on, and break ties toward the earliest.
 */
export async function chooseCandidate(
  candidates: (string | null)[],
  run: Runner,
): Promise<Vote | null> {
  const runs: { index: number; sig: string; empty: boolean }[] = [];
  for (let index = 0; index < candidates.length; index++) {
    const sql = candidates[index];
    if (!sql) continue;
    const result = await run(sql);
    if ("rows" in result) {
      runs.push({ index, sig: signature(result.rows), empty: result.rows.length === 0 });
    }
  }
  if (runs.length === 0) {
    const first = candidates.findIndex((sql) => !!sql);
    if (first < 0) return null;
    return { index: first, sql: candidates[first] as string, agree: 0, ran: 0 };
  }
  const pool = runs.filter((r) => !r.empty);
  const voters = pool.length > 0 ? pool : runs;
  const votes = new Map<string, number>();
  for (const r of voters) votes.set(r.sig, (votes.get(r.sig) ?? 0) + 1);
  const best = Math.max(...votes.values());
  const winner = voters.find((r) => votes.get(r.sig) === best) as { index: number };
  const sql = candidates[winner.index] as string;
  return { index: winner.index, sql, agree: best, ran: runs.length };
}