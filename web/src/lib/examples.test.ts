import { describe, expect, it } from "vitest";

import { findExample } from "./examples";

describe("findExample", () => {
  it("matches ignoring case and trailing punctuation", () => {
    expect(findExample("how many singers are older than 30")?.sql).toContain("COUNT(*)");
    expect(findExample("  Show the names of singers from France?  ")).toBeDefined();
  });

  it("returns nothing for other questions", () => {
    expect(findExample("What is the oldest singer?")).toBeUndefined();
  });
});