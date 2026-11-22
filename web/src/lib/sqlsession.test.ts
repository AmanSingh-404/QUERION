import { afterEach, describe, expect, it, vi } from "vitest";

import { SqlSession, type Table } from "./sqlsession";

type Message = { id: number; type: string; payload: { sql?: string } };

class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  terminated = false;

  constructor() {
    FakeWorker.instances.push(this);
  }

  postMessage(message: Message) {
    if (message.type === "run" && message.payload.sql?.includes("hang")) return;
    const data =
      message.type === "load"
        ? { id: message.id, ok: true }
        : { id: message.id, ok: true, result: { columns: ["x"], rows: [[1]] } };
    queueMicrotask(() => this.onmessage?.({ data }));
  }

  terminate() {
    this.terminated = true;
  }
}

const table: Table = { name: "t", columns: [{ name: "x", type: "INTEGER" }], rows: [["1"]] };

afterEach(() => {
  vi.unstubAllGlobals();
  FakeWorker.instances = [];
});

describe("SqlSession", () => {
  it("returns query results from the worker", async () => {
    vi.stubGlobal("Worker", FakeWorker);
    const session = new SqlSession(50);
    await session.load(table);
    expect(await session.run("SELECT 1")).toEqual({ columns: ["x"], rows: [[1]] });
  });

  it("refuses writes without bothering the worker", async () => {
    vi.stubGlobal("Worker", FakeWorker);
    const session = new SqlSession(50);
    await session.load(table);
    expect(await session.run("DROP TABLE t")).toEqual({ error: "only SELECT queries are allowed" });
  });

  it("stops a query that never finishes and stays usable afterwards", async () => {
    vi.stubGlobal("Worker", FakeWorker);
    const session = new SqlSession(50);
    await session.load(table);
    const slow = await session.run("SELECT hang");
    expect(slow).toMatchObject({ error: expect.stringContaining("stopped") });
    expect(FakeWorker.instances).toHaveLength(2);
    expect(FakeWorker.instances[0].terminated).toBe(true);
    expect(await session.run("SELECT 1")).toEqual({ columns: ["x"], rows: [[1]] });
  });
});