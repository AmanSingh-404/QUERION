import { describe, expect, it } from "vitest";

import { type RunResult, chooseCandidate, signature } from "./vote";

describe("signature", () => {
  it("ignores row order and treats 1 and 1.0 alike", () => {
    const a = signature([[1, "a"], [2, "b"]]);
    const b = signature([[2, "b"], [1.0, "a"]]);
    expect(a).toBe(b);
  });

  it("tells numbers from strings and rounds floats", () => {
    expect(signature([[1]])).not.toBe(signature([["1"]]));
    expect(signature([[0.1 + 0.2]])).toBe(signature([[0.3]]));
  });
});

const rows = (...r: unknown[][]): RunResult => ({ rows: r });
const results: Record<string, RunResult> = {
  "SELECT bad": { error: "no such column" },
  "SELECT none": rows(),
  "SELECT a": rows([1], [2]),
  "SELECT a DESC": rows([2], [1]),
  "SELECT one": rows([1]),
};
const run = (sql: string): RunResult => results[sql];

describe("chooseCandidate", () => {
  it("ignores errors and empty results and takes the majority", () => {
    const candidates = ["SELECT bad", "SELECT none", "SELECT one", "SELECT a", "SELECT a DESC"];
    const picked = chooseCandidate(candidates, run);
    expect(picked).toEqual({ index: 3, sql: "SELECT a", agree: 2, ran: 4 });
  });

  it("breaks ties toward the earliest candidate", () => {
    expect(chooseCandidate(["SELECT one", "SELECT a"], run)?.index).toBe(0);
  });

  it("falls back to the first candidate when nothing runs, and to null with none", () => {
    const fallback = chooseCandidate(["SELECT bad", null], run);
    expect(fallback).toEqual({ index: 0, sql: "SELECT bad", agree: 0, ran: 0 });
    expect(chooseCandidate([null, null], run)).toBeNull();
  });
});