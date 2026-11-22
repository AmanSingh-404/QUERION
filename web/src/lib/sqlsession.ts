import { type Column, createStatement } from "./schema";
import { type Cell, checkSql, convertCell, insertStatement } from "./sqlcheck";

export type Table = { name: string; columns: Column[]; rows: string[][] };
export type QueryResult = { columns: string[]; rows: unknown[][] } | { error: string };

type Reply = {
  id: number;
  ok: boolean;
  result?: { columns: string[]; rows: unknown[][] };
  error?: string;
};
type Payload = { create: string; insert: string; rows: Cell[][] };

const LOAD_TIMEOUT_MS = 20000;

class TimeoutError extends Error {}

/**
 * SQLite in a Web Worker. Queries run one at a time with a time limit; a query that overruns
 * it is stopped by replacing the worker and reloading the table, so the page never freezes.
 */
export class SqlSession {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<number, (reply: Reply) => void>();
  private payload: Payload | null = null;
  private tail: Promise<unknown> = Promise.resolve();
  private readonly timeoutMs: number;

  constructor(timeoutMs = 5000) {
    this.timeoutMs = timeoutMs;
  }

  private ensure(): Worker {
    if (!this.worker) {
      const worker = new Worker("/sql-worker.js");
      worker.onmessage = (event: MessageEvent<Reply>) => {
        const done = this.pending.get(event.data.id);
        if (done) {
          this.pending.delete(event.data.id);
          done(event.data);
        }
      };
      worker.onerror = () => {
        for (const [id, done] of this.pending) {
          done({ id, ok: false, error: "the SQL engine failed to start" });
        }
        this.pending.clear();
      };
      this.worker = worker;
    }
    return this.worker;
  }

  private send(type: "load" | "run", payload: unknown, limitMs: number): Promise<Reply> {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new TimeoutError("timed out"));
      }, limitMs);
      this.pending.set(id, (reply) => {
        clearTimeout(timer);
        resolve(reply);
      });
      this.ensure().postMessage({ id, type, payload });
    });
  }

  private async sendLoad(): Promise<void> {
    if (!this.payload) return;
    const reply = await this.send("load", this.payload, LOAD_TIMEOUT_MS);
    if (!reply.ok) throw new Error(reply.error ?? "could not load the table");
  }

  private async restart(): Promise<void> {
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
    await this.sendLoad();
  }

  private enqueue<T>(job: () => Promise<T>): Promise<T> {
    const next = this.tail.then(job, job);
    this.tail = next.catch(() => undefined);
    return next;
  }

  load(table: Table): Promise<void> {
    this.payload = {
      create: createStatement(table.name, table.columns),
      insert: insertStatement(table.name, table.columns.length),
      rows: table.rows.map((row) => table.columns.map((c, i) => convertCell(row[i], c.type))),
    };
    return this.enqueue(() => this.sendLoad());
  }

  run(sql: string): Promise<QueryResult> {
    const problem = checkSql(sql);
    if (problem) return Promise.resolve({ error: problem });
    return this.enqueue(async () => {
      try {
        const reply = await this.send("run", { sql }, this.timeoutMs);
        return reply.ok && reply.result ? reply.result : { error: reply.error ?? "query failed" };
      } catch (e) {
        if (e instanceof TimeoutError) {
          await this.restart();
          return { error: `The query ran longer than ${this.timeoutMs / 1000} s and was stopped.` };
        }
        throw e;
      }
    });
  }

  close(): void {
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
  }
}