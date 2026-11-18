import { afterEach, describe, expect, it, vi } from "vitest";

import { type GenerateEvent, generate, parseEvent } from "./api";

const body = { tables: ["CREATE TABLE t (a INTEGER)"], question: "q" };

function streamOf(chunks: string[]) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(): Promise<GenerateEvent[]> {
  const seen: GenerateEvent[] = [];
  for await (const event of generate("http://api.test", body)) seen.push(event);
  return seen;
}

afterEach(() => vi.unstubAllGlobals());

describe("parseEvent", () => {
  it("parses candidate and done events and ignores incomplete blocks", () => {
    const candidate = 'event: candidate\ndata: {"index": 2, "sql": "SELECT 1"}';
    expect(parseEvent(candidate)).toEqual({ type: "candidate", index: 2, sql: "SELECT 1" });
    const done = 'event: done\ndata: {"candidates": ["SELECT 1", null], "seconds": 1.2}';
    expect(parseEvent(done)).toEqual({
      type: "done",
      candidates: ["SELECT 1", null],
      seconds: 1.2,
    });
    expect(parseEvent("event: start")).toBeNull();
  });
});

describe("generate", () => {
  it("reassembles events that are split across network chunks", async () => {
    const text =
      'event: start\ndata: {"samples": 1}\n\n' +
      'event: candidate\ndata: {"index": 0, "sql": "SELECT 1"}\n\n';
    const cut = text.indexOf("candidate") + 3;
    const reply = new Response(streamOf([text.slice(0, cut), text.slice(cut)]), { status: 200 });
    vi.stubGlobal("fetch", vi.fn(async () => reply));
    expect((await collect()).map((e) => e.type)).toEqual(["start", "candidate"]);
  });

  it("turns an error response into an exception with the server's message", async () => {
    const reply = new Response(JSON.stringify({ detail: "rate limit exceeded" }), { status: 429 });
    vi.stubGlobal("fetch", vi.fn(async () => reply));
    await expect(collect()).rejects.toThrow("429: rate limit exceeded");
  });
});