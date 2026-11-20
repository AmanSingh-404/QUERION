import { copyFileSync, mkdirSync } from "node:fs";

mkdirSync("public", { recursive: true });
for (const file of ["sql-wasm.js", "sql-wasm.wasm"]) {
  copyFileSync(`node_modules/sql.js/dist/${file}`, `public/${file}`);
}