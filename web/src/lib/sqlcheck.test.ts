import { describe, expect, it } from "vitest";

import { checkSql, convertCell, insertStatement } from "./sqlcheck";

describe("checkSql", () => {
  it("allows one SELECT or WITH statement", () => {
    expect(checkSql("SELECT 1")).toBeNull();
    expect(checkSql("  with a as (select 1) select * from a;")).toBeNull();
  });

  it("refuses writes and several statements", () => {
    expect(checkSql("DROP TABLE t")).not.toBeNull();
    expect(checkSql("SELECT 1; DROP TABLE t")).not.toBeNull();
  });
});

describe("convertCell", () => {
  it("makes NULLs and numbers, and keeps text that is not a number", () => {
    expect(convertCell("", "TEXT")).toBeNull();
    expect(convertCell(undefined, "INTEGER")).toBeNull();
    expect(convertCell("5", "INTEGER")).toBe(5);
    expect(convertCell("1.5", "REAL")).toBe(1.5);
    expect(convertCell("abc", "INTEGER")).toBe("abc");
    expect(convertCell(" x ", "TEXT")).toBe("x");
  });
});

describe("insertStatement", () => {
  it("has one placeholder per column", () => {
    expect(insertStatement("t", 3)).toBe("INSERT INTO t VALUES (?, ?, ?)");
  });
});