import { describe, expect, it } from "vitest";

import { createStatement, inferColumns, safeName } from "./schema";

describe("safeName", () => {
  it("makes SQL-safe names", () => {
    expect(safeName("First Name!", "x")).toBe("First_Name");
    expect(safeName("2024 sales", "x")).toBe("c_2024_sales");
    expect(safeName("   ", "fallback")).toBe("fallback");
    expect(safeName("order", "x")).toBe("order_");
  });
});

describe("inferColumns", () => {
  it("infers integer, real and text columns", () => {
    const rows = [
      ["Ann", "30", "1.5"],
      ["Bob", "25", "2"],
    ];
    expect(inferColumns(["Name", "Age", "Score"], rows)).toEqual([
      { name: "Name", type: "TEXT" },
      { name: "Age", type: "INTEGER" },
      { name: "Score", type: "REAL" },
    ]);
  });

  it("keeps column names unique", () => {
    const columns = inferColumns(["a", "A", "a"], [["1", "2", "3"]]);
    expect(columns.map((c) => c.name)).toEqual(["a", "A_x", "a_x_x"]);
  });
});

describe("createStatement", () => {
  it("writes a single-line CREATE TABLE like the model saw in training", () => {
    const columns = [
      { name: "Name", type: "TEXT" as const },
      { name: "Age", type: "INTEGER" as const },
    ];
    expect(createStatement("people", columns)).toBe("CREATE TABLE people (Name TEXT, Age INTEGER)");
  });
});